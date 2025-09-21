from mmdet.registry import MODELS
from abc import ABCMeta
from mmdet.utils import ConfigType, OptConfigType, InstanceList, OptMultiConfig
from mmengine.model import BaseModule
from mmengine.structures import BaseDataElement
from mmdet.models.roi_heads.roi_extractors import BaseRoIExtractor
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi, bbox_overlaps
from mmdet.structures.bbox.transforms import bbox2roi, scale_boxes
from typing import List, Tuple, Union
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from torchvision.transforms import functional as TF, InterpolationMode
import math
import dgl
from .modules.layers import build_mlp
from .modules.gnn import GNNHead

from collections import Counter #---
from collections import defaultdict #---
@MODELS.register_module()
class GraphHead_ssg201(BaseModule, metaclass=ABCMeta):
    """Graph Head to construct graph from detections

    Args:
        edges_per_node (int)
        viz_feat_size (int)
        roi_extractor
        gnn_cfg (ConfigType): gnn cfg
    """
    def __init__(self, edges_per_node: int, viz_feat_size: int, num_triplet_edge_classes: int, #---
            roi_extractor: BaseRoIExtractor, num_edge_classes: int,
            presence_loss_cfg: ConfigType, presence_loss_weight: float,
            classifier_loss_cfg: ConfigType, classifier_loss_weight: float,
             triplet_loss_weight: float,  #---
              hand_loss_weight: float,  #---
            gt_use_pred_detections: bool = False, sem_feat_hidden_dim: int = 2048,
            semantic_feat_projector_layers: int = 3, num_roi_feat_maps: int = 4,
            allow_same_label_edge: List = [5], gnn_cfg: ConfigType = None,
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.triplet_loss_weight = triplet_loss_weight  #---
        self.num_triplet_edge_classes = num_triplet_edge_classes  #---
        self.hand_loss_weight = hand_loss_weight  #---
        # attributes for building graph from detections
        self.edges_per_node = edges_per_node
        self.viz_feat_size = viz_feat_size
        self.roi_extractor = roi_extractor
        self.num_roi_feat_maps = num_roi_feat_maps
        dim_list = [viz_feat_size, 64, 64]
        self.edge_mlp_sbj = build_mlp(dim_list, batch_norm='batch',
                final_nonlinearity=False)
        self.edge_mlp_obj = build_mlp(dim_list, batch_norm='batch',
                final_nonlinearity=False)
        self.gt_use_pred_detections = gt_use_pred_detections
        self.allow_same_label_edge = torch.tensor(allow_same_label_edge)

        # presence loss
        self.presence_loss = MODELS.build(presence_loss_cfg)
        self.presence_loss_weight = presence_loss_weight
 
        # edge classifier loss
        self.classifier_loss = MODELS.build(classifier_loss_cfg)
        self.classifier_loss_weight = classifier_loss_weight
      
        # gnn attributes
        if gnn_cfg is not None:
            gnn_cfg.input_dim_node = viz_feat_size
            gnn_cfg.input_dim_edge = viz_feat_size
            self.gnn = MODELS.build(gnn_cfg)
        else:
            self.gnn = None

        # attributes for predicting relation class
        self.num_edge_classes = num_edge_classes
        dim_list = [viz_feat_size, viz_feat_size, self.num_edge_classes + 1] # predict no edge or which class
        self.edge_predictor = build_mlp(dim_list, batch_norm='batch', final_nonlinearity=False)
        
        # self.num_edge_classes = num_edge_classes 
        # dim_list = [viz_feat_size, viz_feat_size, self.num_edge_classes + 1] # predict no edge or which class
        # self.edge_predictor = build_mlp(dim_list, batch_norm='batch', final_nonlinearity=False)
        dim_list_verb = [viz_feat_size, viz_feat_size, self.num_triplet_edge_classes + 2] 
        # need add another class null_verb# predict no edge or which class
        self.edge_predictor_verb = build_mlp(dim_list_verb, batch_norm='batch', final_nonlinearity=False)
        self.hand_node_classes = 3
        # Build a self‑edge predictor using your MLP builder
        self.hand_node_predictor = build_mlp(
            [viz_feat_size, viz_feat_size, self.hand_node_classes],
            batch_norm='batch',
            final_nonlinearity=False
        )
        # make query projector if roi_extractor is None
        if self.roi_extractor is None:
            dim_list = [viz_feat_size] * 3
            self.edge_query_projector = build_mlp(dim_list, batch_norm='batch')

    def _predict_edge_presence(self, node_features, nodes_per_img):
        # EDGE PREDICTION
        mlp_input = node_features.flatten(end_dim=1)
        if mlp_input.shape[0] == 1:
            sbj_feats = self.edge_mlp_sbj(torch.cat([mlp_input, mlp_input]))[0].unsqueeze(0)
        else:
            sbj_feats = self.edge_mlp_sbj(mlp_input)
        if mlp_input.shape[0] == 1:
            obj_feats = self.edge_mlp_obj(torch.cat([mlp_input, mlp_input]))[0].unsqueeze(0)
        else:
            obj_feats = self.edge_mlp_obj(mlp_input)

        sbj_feats = sbj_feats.view(len(node_features), -1,
                sbj_feats.size(-1)) # B x N x F, where F is feature dimension
        obj_feats = obj_feats.view(len(node_features), -1,
                obj_feats.size(-1)) # B x N x F, where F is feature dimension

        # get likelihood of edge between each pair of proposals using kernel
        edge_presence_logits = torch.bmm(sbj_feats, obj_feats.transpose(1, 2)) # B x N x N
        
        # mask using nodes_per_img
        mask = torch.zeros_like(edge_presence_logits)
        for i, num_nodes in enumerate(nodes_per_img):
            mask[i, num_nodes:, :] = float('-inf')  # Set mask for subject nodes beyond the number of nodes in the image to -inf
            mask[i, :, num_nodes:] = float('-inf')  # Set mask for object nodes beyond the number of nodes in the image to -inf

        # also mask diagonal
        mask = mask + (torch.eye(mask.shape[1]).unsqueeze(0).to(mask.device) * float('-inf')).nan_to_num(0)

        edge_presence_masked = edge_presence_logits + mask

        return edge_presence_logits, edge_presence_masked

    def _build_edges(self, results: SampleList, nodes_per_img: List, feats: BaseDataElement = None) -> SampleList:
        # get boxes, rescale
        scale_factor = results[0].scale_factor
        boxes = pad_sequence([r.pred_instances.bboxes for r in results], batch_first=True)
        boxes_per_img = [len(r.pred_instances.bboxes) for r in results]
        rescaled_boxes = scale_boxes(boxes.float(), scale_factor)

        # compute all box_unions
        edge_boxes = self.box_union(rescaled_boxes, rescaled_boxes)

        # select valid edge boxes
        valid_edge_boxes = [e[:b, :b].flatten(end_dim=1) for e, b in zip(edge_boxes, boxes_per_img)]
        edge_rois = bbox2roi(valid_edge_boxes)
    
        # compute edge feats
        if self.roi_extractor is not None:
            roi_input_feats = feats.neck_feats[:self.num_roi_feat_maps] \
                    if feats.neck_feats is not None else feats.bb_feats[:self.num_roi_feat_maps]
            edge_viz_feats = self.roi_extractor(roi_input_feats, edge_rois).squeeze(-1).squeeze(-1)
        else:
            # offset is just cum sum of edges_per_img
            edges_per_img = torch.tensor([b*b for b in boxes_per_img])
            edge_offsets = torch.cat([torch.zeros(1), torch.cumsum(edges_per_img, 0)[:-1]]).repeat_interleave(edges_per_img)

            # define edges to keep
            edges_to_keep = (torch.cat([torch.arange(e) for e in edges_per_img]) + edge_offsets).to(boxes.device)

            # densely add object queries to get edge feats, select edges with edges_to_keep
            edge_viz_feats = self.edge_query_projector((feats.instance_feats.unsqueeze(1) + \
                    feats.instance_feats.unsqueeze(2)).flatten(end_dim=-2))[edges_to_keep.long()]

        # predict edge presence
        edge_presence_logits, edge_presence_masked = self._predict_edge_presence(feats.instance_feats, nodes_per_img)

        # collate all edge information
        edges = BaseDataElement()
        edges.boxes = torch.cat(valid_edge_boxes)
        edges.boxesA = torch.cat([b[:num_b].repeat_interleave(num_b, dim=0) for num_b, b in zip(
                boxes_per_img, rescaled_boxes)])
        edges.boxesB = torch.cat([b[:num_b].repeat(num_b, 1) for num_b, b in zip(
                boxes_per_img, rescaled_boxes)])
        edges.edges_per_img = [num_b * num_b for num_b in boxes_per_img]
        edges.viz_feats = edge_viz_feats

        # store presence logits
        edges.presence_logits = edge_presence_masked

        return edges, edge_presence_logits

    def _predict_edge_classes2(self, graph: BaseDataElement, batch_input_shape: tuple) -> BaseDataElement:
        # predict edge class
        # this is the original code without oper used for validaiton  or test 
        edge_predictor_input = graph.edges.viz_feats + graph.edges.gnn_viz_feats
        if edge_predictor_input.shape[0] == 1:
            graph.edges.class_logits = self.edge_predictor(edge_predictor_input.repeat(2, 1))[0].unsqueeze(0)
            graph.edges.class_logits_verb = self.edge_predictor_verb(edge_predictor_input.repeat(2, 1))[0].unsqueeze(0)
        else:
            graph.edges.class_logits = self.edge_predictor(edge_predictor_input)
            graph.edges.class_logits_verb = self.edge_predictor_verb(edge_predictor_input)
            

        return graph
    
    
    def _predict_edge_classes(self, graph: BaseDataElement, batch_input_shape: tuple) -> BaseDataElement:
        # Predict edge classes, verb classes, and node classes for the graph.

        # The graph has two main components: 'edges' and 'nodes'.
        # 'nodes' contain features like 'viz_feats', 'gnn_viz_feats', and 'nodes_per_img'.
        # 'edges' contain features such as 'viz_feats', 'gnn_viz_feats', and additional metadata.

        # Combine the original edge visual features with the GNN-updated edge features.
        edge_predictor_input = graph.edges.viz_feats + graph.edges.gnn_viz_feats  # Shape: [num_edges, feature_dim]

        # Handle the case where there is only one edge to avoid dimension issues.
        if edge_predictor_input.shape[0] == 1:
            # Repeat the single edge feature to create a pseudo-batch of two,
            # then pass it through the edge predictor to obtain class logits.
            graph.edges.class_logits = self.edge_predictor(edge_predictor_input.repeat(2, 1))[0].unsqueeze(0)
            # Similarly, compute the verb class logits using the verb edge predictor.
            graph.edges.class_logits_verb = self.edge_predictor_verb(edge_predictor_input.repeat(2, 1))[0].unsqueeze(0)
        else:
            # For multiple edges, directly compute the class logits.
            graph.edges.class_logits = self.edge_predictor(edge_predictor_input)
            # Compute the verb class logits for all edges.
            graph.edges.class_logits_verb = self.edge_predictor_verb(edge_predictor_input)

        # Retrieve the device (CPU or GPU) where the node GNN features are located.
        device = graph.nodes.gnn_viz_feats.device
        # Ensure the node visual features are on the same device.
        #=======================================================
        # node features 
        graph.nodes.viz_feats = graph.nodes.viz_feats.to(device)
        #=======================================================
        # Combine the node's GNN-updated features with its original visual features.
        node_features = graph.nodes.gnn_viz_feats + graph.nodes.viz_feats  # Expected shape: [B, N, D]

        # Unpack the dimensions: B = batch size, N = number of nodes per image, D = feature dimension.
        B, N, D = node_features.shape

        # If there are no nodes (N == 0), create an empty tensor with the expected output shape.
        if N == 0:
            output = torch.empty(B, 0, self.hand_node_classes, device=device)
        else:
            # Flatten the node features from shape [B, N, D] to [B * N, D] for processing.
            node_features = node_features.view(B * N, D)
            # Pass the flattened node features through the hand node predictor to compute node logits.
            output = self.hand_node_predictor(node_features)
            # Reshape the output back to [B, N, hand_node_classes] to match the original batch structure.
            output = output.view(B, N, self.hand_node_classes)

        # Store the computed node logits in the graph under the key 'node_logit' in the edges structure.
        graph.edges.node_logit = output

        # Return the updated graph containing edge class logits, verb logits, and node logits.
        return graph

    def _select_edges(self, edges: BaseDataElement, nodes_per_img: List) -> BaseDataElement:
        # SELECT TOP E EDGES PER NODE USING PRESENCE LOGITS, COMPUTE EDGE FLATS
        presence_logits = edges.presence_logits # B x N x N
     
        # Assuming edges.boxesA contains class labels
        # batch_0_classes = [(idx, edges.boxesA[idx].item()) for idx_pair in batch_0_edges for idx in idx_pair]
        # pick top E edges per node, or N - 1 if num nodes is too small
        edge_flats, edge_indices = self._edge_flats_from_adj_mat(presence_logits, nodes_per_img)
        edges_per_img = Tensor([len(ef) for ef in edge_flats]).to(presence_logits.device).int()

        edges.edges_per_img = edges_per_img
        edges.batch_index = torch.arange(len(edges_per_img)).to(edges_per_img.device).repeat_interleave(
                edges_per_img).view(-1, 1) # stores the batch_id of each edge
        edges.edge_flats = torch.cat([edges.batch_index, torch.cat(edge_flats)], dim=1)
        edges.boxes = edges.boxes[edge_indices]
        edges.boxesA = edges.boxesA[edge_indices]
        edges.boxesB = edges.boxesB[edge_indices]
        edges.viz_feats = edges.viz_feats[edge_indices]
        # print('.edge_flats', edge_flats)
        # print("Total edges before filtering:", original_edge_indices.shape[0])
       
        return edges


    def _edge_flats_from_adj_mat(self, presence_logits, nodes_per_img):
        edge_flats = []
        edge_indices = []
        edge_index_offset = 0
        num_edges = torch.minimum(torch.ones(len(presence_logits)) * self.edges_per_node,
                Tensor(nodes_per_img) - 1).int()

        for pl, ne, nn in zip(presence_logits, num_edges, nodes_per_img):
            if nn == 0:
                edge_flats.append(torch.zeros(0, 2).to(pl.device).int())
                edge_indices.append(torch.zeros(0).to(pl.device))
                continue

            row_indices = torch.arange(nn).to(pl.device).view(-1, 1).repeat(1, ne.item())
            topk_indices = torch.topk(pl, k=ne.item(), dim=1).indices
            edge_flat = torch.stack([row_indices, topk_indices[:nn]], dim=-1).long().view(-1, 2)
            edge_flat = edge_flat[self.drop_duplicates(edge_flat.sort(dim=1).values).long()]
            edge_indices.append(torch.arange(nn * nn).to(pl.device).view(nn, nn)[edge_flat[:, 0], edge_flat[:, 1]] + \
                    edge_index_offset)
            edge_index_offset = edge_index_offset + nn * nn
            edge_flats.append(edge_flat)

        return edge_flats, torch.cat(edge_indices).long()

    def predict(self, results: SampleList, feats: BaseDataElement) -> Tuple[BaseDataElement]:
        nodes_per_img = [len(r.pred_instances.bboxes) for r in results]
        print('readin this predict part of the code in ssg-com')
        # assert print('stopppp')
        # build edges for GT
        gt_edges = self._build_gt_edges(results)
        # Add triplet edges (verb ground truth) - this was missing!
        gt_edges = self._build_triplet_edges(results, gt_edges)
        
        # build edges
        edges, _ = self._build_edges(results, nodes_per_img, feats)

        # select edges
        edges = self._select_edges(edges, nodes_per_img)

        # construct graph out of edges and result
        graph = self._construct_graph(feats, edges, nodes_per_img)

        # apply gnn
        if self.gnn is not None:
            dgl_g = self.gnn(graph)

            # update graph
            graph = self._update_graph(graph, dgl_g)
        #=================================
        pred_bboxes = [x.pred_instances.bboxes for x in results]
        graph.edges.pred_node_boxes = pred_bboxes
        #=================================
        # print('graph.edges keys : ', graph.edges.keys())
        # print('graph.nodes keys : ', graph.nodes.keys())
        # predict edge classes
        # graph = self._predict_edge_classes2(graph, results[0].batch_input_shape)
        graph = self._predict_edge_classes(graph, results[0].batch_input_shape)
        # print('graph keys : ', graph.keys())
        # print('graph.edges keys : ', graph.edges.keys())
        # print('graph.nodes keys : ', graph.nodes.keys())
        # print('gt_edges keys : ', gt_edges.keys())
        # print('graph.nodes.class_logits_verb : ', graph.edges.class_logits_verb.shape)
        # print('graph.nodes.node_logit : ', graph.edges.node_logit.shape)
        # print('gt_edges._verb_edge_relations:', len(gt_edges._verb_edge_relations))

        # print('gt_edges._verb_edge_relations : ', gt_edges._verb_edge_relations[0].shape)
        # print('gt_edges.oper_labels_per_batch : ', gt_edges.oper_labels_per_batch.shape)
        # assert print('stopppp')
        # # Add verb predictions to results so they can be accessed later
        # for i, result in enumerate(results):
        #     # Get the edges for this image
        #     start_idx = sum(graph.edges.edges_per_img[:i]) if i > 0 else 0
        #     end_idx = start_idx + graph.edges.edges_per_img[i]
            
        #     # Extract verb predictions for this image
        #     if hasattr(graph.edges, 'class_logits_verb') and graph.edges.class_logits_verb.numel() > 0:
        #         verb_logits = graph.edges.class_logits_verb[start_idx:end_idx]
        #         verb_predictions = torch.argmax(verb_logits, dim=1)
        #         verb_scores = torch.softmax(verb_logits, dim=1)
                
        #         # Add to result
        #         result.verb_predictions = verb_predictions
        #         result.verb_scores = verb_scores
        #         result.verb_logits = verb_logits
        #     else:
        #         # Empty verb predictions if no edges
        #         result.verb_predictions = torch.empty(0, dtype=torch.long, device=result.pred_instances.bboxes.device)
        #         result.verb_scores = torch.empty(0, dtype=torch.float, device=result.pred_instances.bboxes.device)
        #         result.verb_logits = torch.empty(0, dtype=torch.float, device=result.pred_instances.bboxes.device)
            
        #     # Extract hand node predictions for this image
        #     if hasattr(graph.edges, 'node_logit') and graph.edges.node_logit.numel() > 0:
        #         node_logits = graph.edges.node_logit[i]  # [num_nodes, num_hand_classes]
        #         node_predictions = torch.argmax(node_logits, dim=1)
        #         node_scores = torch.softmax(node_logits, dim=1)
                
        #         # Add to result
        #         result.hand_node_predictions = node_predictions
        #         result.hand_node_scores = node_scores
        #         result.hand_node_logits = node_logits
        #     else:
        #         # Empty hand node predictions if no nodes
        #         result.hand_node_predictions = torch.empty(0, dtype=torch.long, device=result.pred_instances.bboxes.device)
        #         result.hand_node_scores = torch.empty(0, dtype=torch.float, device=result.pred_instances.bboxes.device)
        #         result.hand_node_logits = torch.empty(0, dtype=torch.float, device=result.pred_instances.bboxes.device)

        return graph, gt_edges

    def _construct_graph(self, feats: BaseDataElement, edges: BaseDataElement,
                nodes_per_img: List) -> BaseDataElement:
        graph = BaseDataElement()
        graph.edges = edges

        # move result data into nodes
        nodes = BaseDataElement()
        nodes.viz_feats = feats.instance_feats
        nodes.nodes_per_img = nodes_per_img
        graph.nodes = nodes

        return graph

    def _update_graph(self, graph: BaseDataElement, dgl_g: dgl.DGLGraph) -> BaseDataElement:
        device = dgl_g.device  # 현재 DGL 그래프가 위치한 디바이스 가져오기
        
        # update node viz feats (leave semantic feats the same, add to original feats)
        updated_node_feats = pad_sequence(dgl_g.ndata['viz_feats'].split(graph.nodes.nodes_per_img),
                batch_first=True)
        graph.nodes.gnn_viz_feats = updated_node_feats

        # update graph structure
        graph.edges.edges_per_img = dgl_g.batch_num_edges()
        graph.edges.batch_index = torch.arange(len(graph.edges.edges_per_img)).to(
                graph.edges.edges_per_img.device).repeat_interleave(graph.edges.edges_per_img).view(-1, 1)

        # batch_edge_offset = torch.cat([torch.zeros(1),
        #         dgl_g.batch_num_nodes()[:-1]], 0).cumsum(0).to(graph.edges.batch_index.device)
        # batch_edge_offset = torch.cat([torch.zeros(1, device=device), dgl_g.batch_num_nodes()[:-1]], 0).cumsum(0) #---
        device = dgl_g.device  # Ensure we get the correct device
        batch_edge_offset = torch.cat([
            torch.zeros(1, device=device),                     # Tensor on GPU (cuda:0)
            dgl_g.batch_num_nodes()[:-1].to(device)            # Move to the same device
        ], dim=0).cumsum(0)                                   # Apply cumulative sum
        
        edge_flats = torch.stack(dgl_g.edges(), 1) - \
                batch_edge_offset[graph.edges.batch_index].view(-1, 1)
        graph.edges.edge_flats = torch.cat([graph.edges.batch_index, edge_flats], 1).long()

        # update edge data (skip connection to orig edge feats)
        graph.edges.boxes = dgl_g.edata['boxes'].split(graph.edges.edges_per_img.tolist())
        graph.edges.boxesA = dgl_g.edata['boxesA'].split(graph.edges.edges_per_img.tolist())
        graph.edges.boxesB = dgl_g.edata['boxesB'].split(graph.edges.edges_per_img.tolist())
        graph.edges.gnn_viz_feats = dgl_g.edata['gnn_feats']

        return graph

    def _build_gt_edges(self, results: SampleList) -> BaseDataElement:
        boxes_per_img = []
        bounding_boxes = []
        all_labels = []
        is_gt_box = []  # ✅ GT 박스 여부 저장
        
        for r in results:
            if r.is_det_keyframe and not self.gt_use_pred_detections:
                boxes = r.gt_instances.bboxes # ex: torch.Size([N, 4]) => N: box class, 4: Coordinates 
                labels = r.gt_instances.labels # ex: torch.Size([N]) => N: box class 
                is_gt_box.append(torch.ones(len(boxes), dtype=torch.bool))  # ✅ GT 박스 → True  #---
                
                bounding_boxes.append(boxes)
                all_labels.append(labels)
                boxes_per_img.append(len(boxes))

            else:
                # get boxes, rescale
                boxes = r.pred_instances.bboxes
                scores = r.pred_instances.scores
                labels = r.pred_instances.labels

                # use score thresh 0.3 to filter boxes
                boxes = boxes[scores > 0.3]
                labels = labels[scores > 0.3]

                # convert to tensor and scale boxes
                bounding_boxes.append(scale_boxes(boxes.float(), r.scale_factor))
                boxes_per_img.append(len(bounding_boxes))
                all_labels.append(labels)

                is_gt_box.append(torch.zeros(len(boxes), dtype=torch.bool))  # ✅ pseudo 박스 → False  #---
        
        '''        
        ✅ convert to tensor (list => tensor)
        1. before pad_sequence
        bounding_boxes: [torch.Size([4, 4]), torch.Size([5, 4]), torch.Size([6, 4]), torch.Size([5, 4]), torch.Size([4, 4]), torch.Size([4, 4]), torch.Size([3, 4]), torch.Size([2, 4])]
        all_labels: [torch.Size([4]), torch.Size([5]), torch.Size([6]), torch.Size([5]), torch.Size([4]), torch.Size([4]), torch.Size([3]), torch.Size([2])]
        is_gt_box: [torch.Size([4]), torch.Size([5]), torch.Size([6]), torch.Size([5]), torch.Size([4]), torch.Size([4]), torch.Size([3]), torch.Size([2])]
        
        1. after pad_sequence
        bounding_boxes: torch.Size([8, 6, 4]) => [torch.Size([6, 4]), torch.Size([6, 4]), ..., torch.Size([6, 4])]
        all_labels: torch.Size([8, 6]) => [torch.Size([6]), torch.Size([6]), ..., torch.Size([6])]
        is_gt_box: torch.Size([8, 6]) => [torch.Size([6]), torch.Size([6]), ..., torch.Size([6])]
        len(bounding_boxes), len(all_labels), len(is_gt_box) : 8(batch size)
        '''
        bounding_boxes = pad_sequence(bounding_boxes, batch_first=True)
        all_labels = pad_sequence(all_labels, batch_first=True)
        is_gt_box = pad_sequence(is_gt_box, batch_first=True) #---
        
        # compute centroids and distances for general use
        centroids = (bounding_boxes[:, :, :2] + bounding_boxes[:, :, 2:]) / 2
        distance_x = centroids[:, :, 0].unsqueeze(-1) - centroids[:, :, 0].unsqueeze(-2)
        distance_y = centroids[:, :, 1].unsqueeze(-1) - centroids[:, :, 1].unsqueeze(-2)

        relationships = []

        # FIRST COMPUTE INSIDE-OUTSIDE MASK

        # compute areas of all boxes and create meshgrid
        B, N, _ = bounding_boxes.shape
        areas = self.box_area(bounding_boxes) # B x N x 1
        areas_x = areas.unsqueeze(-1).expand(B, N, N)
        areas_y = areas.unsqueeze(-2).expand(B, N, N)

        # compute intersection
        intersection = self.box_intersection(bounding_boxes, bounding_boxes) # B x N x N

        # inside-outside is when intersection is close to the area of the smaller box
        inside_outside_matrix = intersection / torch.minimum(areas_x, areas_y)
        inside_outside_mask = (inside_outside_matrix >= 0.8)

        # COMPUTE LEFT-RIGHT, ABOVE-BELOW, INSIDE-OUTSIDE MASKS

        # compute angle matrix using distance x and distance y
        angle_matrix = torch.atan2(distance_y, distance_x)
        left_right_mask = ((angle_matrix > (-math.pi / 4)) & (angle_matrix <= (math.pi / 4))) | \
                ((angle_matrix > (3 * math.pi / 4)) | (angle_matrix <= (-3 * math.pi / 4)))
        above_below_mask = ((angle_matrix > (math.pi / 4)) & (angle_matrix <= (3 * math.pi / 4))) | \
                ((angle_matrix > (-3 * math.pi / 4)) & (angle_matrix <= (-math.pi / 4)))

        # left right and above below are only when inside outside is False
        left_right_mask = left_right_mask.int() * (~inside_outside_mask).int() # 1 for left-right
        above_below_mask = above_below_mask.int() * (~inside_outside_mask).int() * 2 # 2 for above-below
        inside_outside_mask = inside_outside_mask.int() * 3 # 3 for inside-outside

        relationships = (left_right_mask + above_below_mask + inside_outside_mask).long()

        # SELECT E EDGES PER NODE BASED ON gIoU
        iou_matrix = bbox_overlaps(bounding_boxes, bounding_boxes)

        # mask diagonal, invalid bbox edges
        diag_mask = torch.eye(iou_matrix.shape[-1]).repeat(iou_matrix.shape[0], 1, 1).bool()
        iou_matrix[diag_mask] = float('-inf') # set diagonal to -inf
        iou_matrix[torch.minimum(areas_x, areas_y) == 0] = float('-inf') # set all entries where bbox area is 0 to -inf

        # mask edges between same class
        same_class = (all_labels.unsqueeze(1) == all_labels.unsqueeze(2))
        allow_same_label_edge = self.allow_same_label_edge.view(1, 1, -1).to(all_labels.device)
        same_label_edge_mask = (all_labels.unsqueeze(-1) == allow_same_label_edge).any(-1)
        same_class[same_label_edge_mask] = False
        same_class.permute(0, 2, 1)[same_label_edge_mask] = False
        iou_matrix[same_class] = float('-inf')

        valid_nodes_per_img = [(a > 0).sum().item() for a in areas]
        num_edges_per_img = [min(self.edges_per_node, max(0, v-1)) for v in valid_nodes_per_img] # limit edges based on number of valid boxes per img
        selected_edges = [torch.topk(iou_mat[:v, :v], e, dim=1).indices for iou_mat, v, e in zip(iou_matrix, valid_nodes_per_img, num_edges_per_img)]

        # COMPUTE EDGE FLATS (PAIRS OF OBJECT IDS CORRESPONDING TO EACH EDGE)
        edge_flats = []
        for s in selected_edges:
            # edge flats is just arange, each column of selected edges
            ef = torch.stack([torch.arange(s.shape[0]).view(-1, 1).repeat(1, s.shape[1]).to(s.device),
                s], -1).view(-1, 2)

            # DROP DUPLICATES
            ef = ef[self.drop_duplicates(ef.sort(dim=1).values).long()]

            edge_flats.append(ef)

        # COMPUTE EDGE BOXES AND SELECT USING EDGE FLATS
        edge_boxes = self.box_union(bounding_boxes, bounding_boxes)
        selected_edge_boxes = [eb[ef[:, 0], ef[:, 1]] for eb, ef in zip(edge_boxes, edge_flats)]
        selected_boxesA = [b[ef[:, 0]] for b, ef in zip(bounding_boxes, edge_flats)]
        selected_boxesB = [b[ef[:, 1]] for b, ef in zip(bounding_boxes, edge_flats)]

        # SELECT RELATIONSHIPS USING EDGE FLATS
        selected_relations = [er[ef[:, 0], ef[:, 1]] for er, ef in zip(relationships, edge_flats)]

        
        '''
        edge_flats: [torch.Size([14, 2]), torch.Size([14, 2]), ..., torch.Size([18, 2])] = len: 8
        selected_edge_boxes: [torch.Size([14, 4]), torch.Size([14, 4]), ..., torch.Size([18, 4])] = len: 8
        selected_boxesA: [torch.Size([14, 4]), torch.Size([14, 4]), ..., torch.Size([18, 4])] = len: 8
        selected_boxesB: [torch.Size([14, 4]), torch.Size([14, 4]), ..., torch.Size([18, 4])] = len: 8
        selected_relations: [torch.Size([14]), torch.Size([14]), ..., torch.Size([18])]
        is_gt_box: torch.Size([8, 7])
        all_labels: torch.Size([8, 7])
        '''
        
        # add edge flats, boxes, and relationships to gt_graph structure
        gt_edges = BaseDataElement()
        gt_edges.edge_flats = edge_flats
        gt_edges.edge_boxes = selected_edge_boxes
        gt_edges.boxesA = selected_boxesA
        gt_edges.boxesB = selected_boxesB
        gt_edges.edge_relations = selected_relations
        gt_edges.is_gt_box = is_gt_box  # ✅ 추가된 부분 #--- torch.Size([8, 7])
        gt_edges.all_labels = all_labels  # ✅ GT Labels 추가 #--- torch.Size([8, 7])
        
        return gt_edges

    def drop_duplicates(self, A):
        if A.shape[0] == 0:
            return torch.zeros(0).to(A.device)

        unique, idx, counts = torch.unique(A, dim=0, sorted=True, return_inverse=True,
                return_counts=True)
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0]).to(A.device), cum_sum[:-1]))
        first_indices = ind_sorted[cum_sum]

        return first_indices

    def box_area(self, boxes):
        # boxes: Tensor of shape (batch_size, num_boxes, 4) representing bounding boxes in (x1, y1, x2, y2) format
        width = boxes[..., 2] - boxes[..., 0]  # Compute width
        height = boxes[..., 3] - boxes[..., 1]  # Compute height
        area = width * height  # Compute area

        return area

    def box_union(self, boxes1, boxes2):
        # boxes1, boxes2: Tensors of shape (B, N1, 4) and (B, N2, 4) representing bounding boxes in (x1, y1, x2, y2) format
        B, N1, _ = boxes1.shape
        B, N2, _ = boxes2.shape

        # Expand dimensions to perform broadcasting
        boxes1 = boxes1.unsqueeze(2)  # (B, N1, 1, 4)
        boxes2 = boxes2.unsqueeze(1)  # (B, 1, N2, 4)

        # Compute the coordinates of the intersection bounding boxes
        union_x1 = torch.min(boxes1[:, :, :, 0], boxes2[:, :, :, 0])  # (B, N1, N2)
        union_y1 = torch.min(boxes1[:, :, :, 1], boxes2[:, :, :, 1])  # (B, N1, N2)
        union_x2 = torch.max(boxes1[:, :, :, 2], boxes2[:, :, :, 2])  # (B, N1, N2)
        union_y2 = torch.max(boxes1[:, :, :, 3], boxes2[:, :, :, 3])  # (B, N1, N2)

        return torch.stack([union_x1, union_y1, union_x2, union_y2], -1)

    def box_intersection(self, boxes1, boxes2):
        # boxes1, boxes2: Tensors of shape (B, N1, 4) and (B, N2, 4) representing bounding boxes in (x1, y1, x2, y2) format
        B, N1, _ = boxes1.shape
        B, N2, _ = boxes2.shape

        # Expand dimensions to perform broadcasting
        boxes1 = boxes1.unsqueeze(2)  # (B, N1, 1, 4)
        boxes2 = boxes2.unsqueeze(1)  # (B, 1, N2, 4)

        # Compute the coordinates of the intersection bounding boxes
        intersection_x1 = torch.max(boxes1[:, :, :, 0], boxes2[:, :, :, 0])  # (B, N1, N2)
        intersection_y1 = torch.max(boxes1[:, :, :, 1], boxes2[:, :, :, 1])  # (B, N1, N2)
        intersection_x2 = torch.min(boxes1[:, :, :, 2], boxes2[:, :, :, 2])  # (B, N1, N2)
        intersection_y2 = torch.min(boxes1[:, :, :, 3], boxes2[:, :, :, 3])  # (B, N1, N2)

        # Compute the areas of the intersection bounding boxes
        intersection_width = torch.clamp(intersection_x2 - intersection_x1, min=0)  # (B, N1, N2)
        intersection_height = torch.clamp(intersection_y2 - intersection_y1, min=0)  # (B, N1, N2)
        intersection_area = intersection_width * intersection_height  # (B, N1, N2)

        return intersection_area


    def _build_triplet_edges(self, results: SampleList, gt_edges: BaseDataElement) -> BaseDataElement:
        """Triplet 정보를 활용한 GT Edge 추가 (Tool → Anatomy)"""

        # TOOL, ANATOMY, VERB CATEGORY ID 정의
        TOOL_CATEGORY_ID = {
    'clipper': 5, 'bipolar': 6, 'grasper': 7,
    'scissors': 8, 'hook': 9, 'irrigator': 10
        }
        ANATOMY_CATEGORY_ID = {
    'cystic_plate': 0, 'calot_triangle': 1,
    'cystic_artery': 2, 'cystic_duct': 3, 'gallbladder': 4
        }
        VERB_CATEGORY_ID = {
            "clip": 1, "dissect": 2, "grasp": 3, 
            "coagulate": 4, "retract": 5
        }
        # Convert TOOL_CATEGORY_ID values to a list
        tool_ids_list = list(TOOL_CATEGORY_ID.values())

        # Convert ANATOMY_CATEGORY_ID values to a list
        anatomy_ids_list = list(ANATOMY_CATEGORY_ID.values())
        triplet_edges_per_batch = []
        triplet_edge_classes_per_batch = []

        # ✅ GT 박스의 category_id 정보 가져오기 (GT 박스만 필터링)
        gt_labels_per_batch = [labels[is_gt] for labels, is_gt in zip(gt_edges.all_labels, gt_edges.is_gt_box)]

        # ✅ Iterate over each batch
        updated_edge_flats = []
        for batch_idx, (edges, gt_labels_batch) in enumerate(zip(gt_edges.edge_flats, results)):
            # Get the category labels for the current batch
            gt_labels_batch = gt_labels_batch.gt_instances.labels

            # Map indices in edge_flats to their respective labels
            edge_labels = torch.tensor(
                [[gt_labels_batch[e[0]].item(), gt_labels_batch[e[1]].item()] for e in edges],
                dtype=torch.int64,
                device=edges.device
            )

            updated_edge_flats.append(edge_labels)  # Store the transformed edge list

        triplets = []
        oper_labels_per_batch = []
        oper_bboxes_per_batch = []
        oper_gt_label_per_batch=[]

        for batch_idx, r in enumerate(results):
            #if hasattr(r, "triplet") and isinstance(r.triplet, list):  # Triplet 정보가 있는 경우만 고려
            if hasattr(r, "triplet") and r.triplet:   
                # assert print('stop here')   
                #===============================================
              
                gt_label = r.gt_instances.labels
                gt_bboxes = r.gt_instances.bboxes
                num_oper = len(r.oper)
                

                # Find tool indices by class ID instead of assuming they're at the end
                tool_indices = []
                for i, label in enumerate(gt_label):
                    if label.item() in tool_ids_list:  # tool_ids_list is already defined above
                        tool_indices.append(i)
                
                # Use found tool indices to get labels and bboxes
                if len(tool_indices) > 0:
                    oper_gt_labels = gt_label[tool_indices]
                    oper_gt_bboxes = gt_bboxes[tool_indices]
                else:
                    # No tools found, create empty tensors
                    device = gt_edges.edge_flats[batch_idx].device
                    oper_gt_labels = torch.empty((0,), dtype=torch.int64, device=device)
                    oper_gt_bboxes = torch.empty((0, 4), dtype=torch.float32, device=device)
                
                # device = gt_edges.edge_flats[batch_idx].device
                if num_oper == 0:
                    assert print('stop here num_oper cannot be 0')
                #     oper_gt_labels = gt_label[-num_oper:]  # Get the last N labels (tools)
                #     oper_gt_bboxes = gt_bboxes[-num_oper:]  # Get the last N bboxes (tools)
                #     print('r.keys:', r.keys())
                # else:
                #     print('r.keys:', r.keys())
                #     print('num_oper', num_oper)
                #     print('gt_label', gt_label)
                #     print('gt_bboxes', gt_bboxes)
                #     assert print('stop here')
                #     oper_gt_labels = torch.empty((0,), dtype=torch.int64, device=device)
                #     oper_gt_bboxes = torch.empty((0, 4), dtype=torch.float32, device=device)
                # extract the operation labels, which are tool labels
                # also extract the bbox from the gt_bboxes
         
                oper_labels = []
                for op in r.oper:
                    op_lower = op.lower()
                    if 'rt' in op_lower:
                        oper_label = 0  # Right hand
                    elif 'lt' in op_lower:
                        oper_label = 1  # Left hand
                    elif 'assi' in op_lower:
                        oper_label = 2  # Assistant
                    else:
                        print(f"⚠️ Unknown operation label: {op}")  # Debugging message
                        assert print('stop error here')
                    oper_labels.append(oper_label)
                # generate oper hand label list
                device = gt_edges.edge_flats[batch_idx].device
                oper_labels_tensor = torch.tensor(oper_labels, dtype=torch.int64, device=device) if oper_labels else torch.empty((0,), dtype=torch.int64, device=device)
                oper_bboxes_tensor = oper_gt_bboxes.to(device) if oper_gt_bboxes.numel() > 0 else torch.empty((0, 4), dtype=torch.float32, device=device)
                oper_labels_per_batch.append(oper_labels_tensor)
                oper_bboxes_per_batch.append(oper_bboxes_tensor)
                oper_gt_label_per_batch.append(oper_gt_labels)
                #===============================================
                triplet_edges = []
                triplet_edge_classes = []
                existing_triplet_edges = set()  # 중복 제거를 위한 Set
                tool_target = []  # Reset per batch!
             
                for ann in r.triplet:  
                    tool = ann.get("tool", None)
                    verb = ann.get("verb", None)
                    target = ann.get("target", None)

                    if tool in TOOL_CATEGORY_ID and target in ANATOMY_CATEGORY_ID and verb in VERB_CATEGORY_ID:
                        tool_id = TOOL_CATEGORY_ID[tool]  
                        target_id = ANATOMY_CATEGORY_ID[target] 
                        verb_id = VERB_CATEGORY_ID[verb]  #--

                        # ✅ GT Labels에서 tool_id와 target_id에 해당하는 index 찾기
                        gt_labels = gt_labels_per_batch[batch_idx]
                        # Obtain all indices for the tool instances matching tool_id
                        tool_idx = (gt_labels == tool_id).nonzero(as_tuple=True)[0]
                        target_idx = (gt_labels == target_id).nonzero(as_tuple=True)[0]

                        # ✅ Tool과 Target이 서로 다른 노드에 있는지 확인
                        for t in tool_idx:
                            for a in target_idx:
                                if t != a:  # 자기 자신과 연결되는 Edge는 생성하지 않음
                                    edge_tuple = tuple(sorted([t.item(), a.item()]))  # 정렬하여 중복 방지
                                    if edge_tuple not in existing_triplet_edges:
                                        triplet_edges.append(list(edge_tuple))  # Edge 연결
                                        triplet_edge_classes.append(verb_id)  # Verb ID 저장
                                        existing_triplet_edges.add(edge_tuple)  # 중복 방지용 Set에 추가

                        tool_target.append((tool_id, verb_id, target_id))

                # ✅ Ensure every batch has an entry in triplets (even if empty)
                triplets.append(tool_target)

                # ✅ batch별로 저장
                if len(triplet_edges) > 0:
                    triplet_edges_per_batch.append(torch.tensor(triplet_edges, dtype=torch.int64, device=gt_edges.edge_flats[batch_idx].device))
                    triplet_edge_classes_per_batch.append(torch.tensor(triplet_edge_classes, dtype=torch.int64, device=gt_edges.edge_flats[batch_idx].device))
                else:
                    triplet_edges_per_batch.append(torch.empty((0, 2), dtype=torch.int64, device=gt_edges.edge_flats[batch_idx].device))
                    triplet_edge_classes_per_batch.append(torch.empty((0,), dtype=torch.int64, device=gt_edges.edge_flats[batch_idx].device))
            else:
                # ✅ If there is no "triplet" attribute or triplet is empty, add an empty list
                triplets.append([])
                device = gt_edges.edge_flats[batch_idx].device
                oper_labels_per_batch.append(torch.empty((0,), dtype=torch.int64, device=device))
                oper_bboxes_per_batch.append(torch.empty((0, 4), dtype=torch.float32, device=device))
                oper_gt_label_per_batch.append(torch.empty((0,), dtype=torch.int64, device=device))

        # ✅ Convert updated_edge_flats into a list of lists of tuples for easy comparison
        updated_edge_lists = [[tuple(edge.tolist()) for edge in batch] for batch in updated_edge_flats]

        # ✅ Initialize a list to store verb lists for all batches
        verb_lists = []

        # ✅ Iterate over all batches
        for batch_idx, updated_edge_list in enumerate(updated_edge_lists):

            # Extract (tool_id, verb_id, target_id) triplets for this batch
            triplet_pairs = [(t[0], t[1], t[2]) for t in triplets[batch_idx]]

            # Initialize a list of zeros with the same length as updated_edge_list
            verb_list = [0] * len(updated_edge_list)

            # Find matching indices and assign the corresponding verb_id
            for idx, edge in enumerate(updated_edge_list):
                for tool_id, verb_id, target_id in triplet_pairs:
                    if edge == (tool_id, target_id) or edge == (target_id, tool_id):
                        verb_list[idx] = verb_id  # Assign verb_id at the matching index
                    elif edge[0] in tool_ids_list and edge[1] in anatomy_ids_list or edge[1] in tool_ids_list and edge[0] in anatomy_ids_list:
                        verb_list[idx] = 6  # 6: assign null verb_id for tool-anatomy pairs

            # Store the verb list for this batch
            verb_lists.append(torch.tensor(verb_list, dtype=torch.int64, device=gt_edges.edge_flats[batch_idx].device))

        # ✅ Store verb lists in `gt_edges._verb_edge_relations`
        gt_edges._verb_edge_relations = verb_lists
        gt_edges.oper_bboxes_per_batch = oper_bboxes_per_batch
        gt_edges.oper_labels_per_batch = oper_labels_per_batch
        gt_edges.oper_gt_label_per_batch = oper_gt_label_per_batch

        return gt_edges
    
 

    # this is the code where every thing happens during trianing 
    def loss_and_predict(self, results: SampleList, feats: BaseDataElement) -> Tuple[SampleList, dict]:
        # init loss dict
        losses = {}
        '''
        type(results): <class 'list'>
        results[0] keys: ['gt_instances', 'ignored_instances', 'pred_instances']
        results[0].gt_instances ['bboxes', 'labels']
        example --
        results[0].gt_instances.bboxes torch.Size([8, 4]) 8 is gt boxes in the frame 
        results[0].gt_instances.labels torch.Size([8]) 8 gt bboxes labels
        --------------------
        results[0].pred_instances ['bboxes', 'feats', 'labels', 'scores']
        results[0].pred_instances.labels: torch.Size([16])
        results[0].pred_instances.bboxes: torch.Size([16, 4])
        results[0].pred_instances.scores: torch.Size([16])
        results[0].pred_instances.feats: torch.Size([16, 256])
        
        type (feats): <class 'mmengine.structures.base_data_element.BaseDataElement'>
        feats keys: ['bb_feats', 'neck_feats', 'instance_feats']
        type(feats.instance_feats): <class 'torch.Tensor'>
        type feats.neck_feats: <class 'tuple'>
        type feats.bb_feats: <class 'tuple'>
        '''
 

        
        # 기존 코드와 동일 (GT Edge 생성)
        gt_edges = self._build_gt_edges(results)
        # 기존 edge들에 verb label 또는 null verb label 추가
        gt_edges = self._build_triplet_edges(results, gt_edges)

        # build edges and compute presence probabilities
        nodes_per_img = [len(r.pred_instances.bboxes) for r in results]
    
        edges, presence_logits = self._build_edges(results, nodes_per_img, feats)
       
        # compute edge presence loss
        edge_presence_loss = self.edge_presence_loss(presence_logits, edges, gt_edges)
        
        # select edges, construct graph, apply gnn, and predict edge classes
        edges = self._select_edges(edges, nodes_per_img)
        # Initialize list to store edge labels for all batches
     
        graph = self._construct_graph(feats, edges, nodes_per_img)
        

        # apply gnn
        if self.gnn is not None:
            dgl_g = self.gnn(graph)

            # update graph
            graph = self._update_graph(graph, dgl_g)
        # extract the detector bboex rediction these align with the bbox feature that goes into the grah nodes 
        # detector bboxes and labels (aligned with node features)
        pred_bboxes  = [x.pred_instances.bboxes for x in results]
        pred_labels  = [x.pred_instances.labels for x in results]

        graph.edges.pred_node_boxes  = pred_bboxes
        graph.edges.pred_node_labels = pred_labels
        
        
        graph = self._predict_edge_classes(graph, results[0].batch_input_shape)
        # print('results[0].pred_instances.labels:', results[0].pred_instances.labels)
        # compute edge classifier loss
        edge_classifier_loss = self.edge_classifier_loss(graph.edges, gt_edges)

        # update losses
        losses.update(edge_presence_loss)
        losses.update(edge_classifier_loss)

        return losses, graph

    def edge_presence_loss(self, presence_logits, edges, gt_edges):
        # first match edge boxes to gt edge boxes
        bA = edges.boxesA.split(edges.edges_per_img)
        bB = edges.boxesB.split(edges.edges_per_img)
        pred_matched_inds, pred_unmatched_inds, _ = self.match_boxes(bA, bB,
                gt_edges.boxesA, gt_edges.boxesB, num=32, iou_threshold=0.5, iou_lower_bound=0.2)

        # assign labels (1 if matched, 0 if unmatched)
        training_inds = [torch.cat([m.view(-1), u.view(-1)]) for m, u in zip(pred_matched_inds, pred_unmatched_inds)]
        flat_edge_relations = []
        for pl, t in zip(presence_logits.flatten(start_dim=1), training_inds):
            flat_edge_relations.append(pl[t])

        flat_edge_relations = torch.cat(flat_edge_relations)
        edge_presence_gt = torch.cat([torch.cat([torch.ones_like(m).view(-1), torch.zeros_like(u).view(-1)]) \
                for m, u in zip(pred_matched_inds, pred_unmatched_inds)])
        
        presence_loss = self.presence_loss(flat_edge_relations, edge_presence_gt).nan_to_num(0) * self.presence_loss_weight

        return {'loss_edge_presence': presence_loss}

    def edge_classifier_loss(self, edges, gt_edges):
   
        # ------------------------------
        # 1) Edge (spatial) matching & loss
        # ------------------------------
        pred_matched_inds, _, gt_matched_inds = self.match_boxes(
            edges.boxesA, edges.boxesB,
            gt_edges.boxesA, gt_edges.boxesB,
            num=16, pos_fraction=0.875, iou_lower_bound=0.2
        )

        flat_edge_classes = torch.cat([
            cl[idx.view(-1)] if idx.numel() > 0 else cl.new_zeros((0, cl.size(-1)))
            for cl, idx in zip(edges.class_logits.split(edges.edges_per_img.tolist()), pred_matched_inds)
        ], dim=0)

        edge_classifier_gt = torch.cat([
            r[idx.view(-1)] if idx.numel() > 0 else r.new_zeros((0,), dtype=r.dtype, device=r.device)
            for r, idx in zip(gt_edges.edge_relations, gt_matched_inds)
        ], dim=0)

        classifier_loss = self.classifier_loss(
            flat_edge_classes, edge_classifier_gt
        ).nan_to_num(0) * self.classifier_loss_weight

        # ------------------------------
        # 2) Verb (triplet) loss
        # ------------------------------
        flat_edge_classes_verb = torch.cat([
            cl[idx.view(-1)] if idx.numel() > 0 else cl.new_zeros((0, cl.size(-1)))
            for cl, idx in zip(edges.class_logits_verb.split(edges.edges_per_img.tolist()), pred_matched_inds)
        ], dim=0)

        edge_classifier_gt_verb = torch.cat([
            r[idx.view(-1)] if idx.numel() > 0 else r.new_zeros((0,), dtype=r.dtype, device=r.device)
            for r, idx in zip(gt_edges._verb_edge_relations, gt_matched_inds)
        ], dim=0)

        triplet_classifier_loss = self.classifier_loss(
            flat_edge_classes_verb, edge_classifier_gt_verb
        ).nan_to_num(0) * self.triplet_loss_weight

        # ------------------------------
        # 3) Hand-node loss (tool-only → match to oper boxes)
        # ------------------------------
        tool_class_ids = {5, 6, 7, 8, 9, 10}

        
        if hasattr(edges, 'pred_node_labels') and edges.pred_node_labels is not None:
            node_labels_per_batch = edges.pred_node_labels  # List[Tensor[N_i]]
        else:
            node_labels_per_batch = [lbl for lbl in gt_edges.all_labels]  # Tensor[B, Nmax] fallback

        
        filtered_node_boxes = []
        node_to_original_mapping = []
        for node_boxes, node_labels in zip(edges.pred_node_boxes, node_labels_per_batch):
            # Ensure 1D labels per image
            labels_img = node_labels
            if labels_img.ndim > 1:
                # If padded tensor (B,Nmax), slice to actual N using node_boxes length
                labels_img = labels_img[:node_boxes.shape[0]]

            if labels_img.numel() == 0 or node_boxes.numel() == 0:
                filtered_node_boxes.append(node_boxes.new_zeros((0, 4)))
                node_to_original_mapping.append(torch.empty((0,), dtype=torch.long, device=node_boxes.device))
                continue

            mask = torch.tensor([int(l.item()) in tool_class_ids for l in labels_img],
                                dtype=torch.bool, device=node_boxes.device)
            filtered_node_boxes.append(node_boxes[mask])
            node_to_original_mapping.append(torch.where(mask)[0])

        # Match filtered tool nodes to oper GT boxes
        pred_matched_inds_self, _, gt_matched_inds_self = self.match_boxes_A_only(
            filtered_node_boxes, gt_edges.oper_bboxes_per_batch,
            num=16, pos_fraction=0.875, iou_lower_bound=0.2
        )

        
        pred_matched_inds_self_orig = []
        for f_inds, back_map in zip(pred_matched_inds_self, node_to_original_mapping):
            if f_inds.numel() == 0 or back_map.numel() == 0:
                pred_matched_inds_self_orig.append(f_inds)  # empty
            else:
                pred_matched_inds_self_orig.append(back_map[f_inds])

        # Aggregate predictions and targets across batch
        aggregated_preds, aggregated_targets = [], []
        # edges.node_logit: [B, N, C_hand]
        for node_logits_img, pred_inds_img, oper_labels_img, gt_inds_img in zip(
            edges.node_logit,            # [N, C]
            pred_matched_inds_self_orig, # idx into N
            gt_edges.oper_labels_per_batch,  # [M]
            gt_matched_inds_self        # idx into M
        ):
            if node_logits_img.numel() == 0:
                continue
            if pred_inds_img.numel() == 0 or gt_inds_img.numel() == 0 or oper_labels_img.numel() == 0:
                continue

            # Pick predictions for matched nodes
            pred_logits = node_logits_img[pred_inds_img]  # [K, C]
            # Map matched GT indices to oper labels
            raw_targets = oper_labels_img[gt_inds_img.view(-1)]  # [K'] (K' could differ from K)

            # Handle size mismatch: if only one GT, broadcast; else keep intersection length
            if pred_logits.size(0) != raw_targets.size(0):
                if raw_targets.numel() == 1:
                    raw_targets = raw_targets.repeat(pred_logits.size(0))
                else:
                    # align to min length to avoid shape errors (rare; occurs if sampler trims asymmetrically)
                    L = min(pred_logits.size(0), raw_targets.size(0))
                    pred_logits = pred_logits[:L]
                    raw_targets = raw_targets[:L]

            aggregated_preds.append(pred_logits)
            aggregated_targets.append(raw_targets)

        if aggregated_preds:
            flat_node_classes = torch.cat(aggregated_preds, dim=0)          # [T, C_hand]
            node_classifier_gt = torch.cat(aggregated_targets, dim=0)        # [T]

            node_cls_loss = self.classifier_loss(
                flat_node_classes, node_classifier_gt
            ).nan_to_num(0) * self.hand_loss_weight
        else:
            # No valid matches this batch
            device = edges.class_logits.device if hasattr(edges.class_logits, 'device') else torch.device('cpu')
            node_cls_loss = torch.tensor(0.0, device=device)

        return {
            'loss_edge_classifier': classifier_loss,
            'loss_triplet_classifier': triplet_classifier_loss,
            'loss_node_classifier': node_cls_loss
        }

    def match_boxes(self, pred_boxes_A, pred_boxes_B, gt_boxes_A, gt_boxes_B, iou_threshold=0.5, iou_lower_bound=0.5, num=50, pos_fraction=0.5):
        # pred_boxes_A: List of tensors of length B, where each tensor has shape (N, 4) representing predicted bounding boxes A in (x1, y1, x2, y2) format
        # pred_boxes_B: List of tensors of length B, where each tensor has shape (N, 4) representing predicted bounding boxes B in (x1, y1, x2, y2) format
        # gt_boxes_A: List of tensors of length B, where each tensor has shape (M, 4) representing ground truth bounding boxes A in (x1, y1, x2, y2) format
        # gt_boxes_B: List of tensors of length B, where each tensor has shape (M, 4) representing ground truth bounding boxes B in (x1, y1, x2, y2) format
        # iou_threshold: IoU threshold for matching
        # iou_lower_bound: Lower bound on IoU for returning unmatched boxes
        B = len(pred_boxes_A)

        pred_matched_indices = []
        pred_unmatched_indices = []
        gt_matched_indices = []

        for b in range(B):
            p_A = pred_boxes_A[b]
            p_B = pred_boxes_B[b]
            g_A = gt_boxes_A[b]
            g_B = gt_boxes_B[b]

            N, _ = p_A.shape
            M, _ = g_A.shape

            # compute overlaps, handle no GT boxes
            if M == 0:
                overlaps = torch.zeros(N, 1).to(p_A.device)
            elif N == 0:
                pred_matched_indices.append(torch.tensor([], dtype=torch.int64, device=p_A.device))
                pred_unmatched_indices.append(torch.tensor([], dtype=torch.int64, device=p_A.device))
                gt_matched_indices.append(torch.tensor([], dtype=torch.int64, device=p_A.device))
                continue
            else:
                overlaps_AA = bbox_overlaps(p_A, g_A)
                overlaps_BB = bbox_overlaps(p_B, g_B)
                overlaps_AB = bbox_overlaps(p_A, g_B)
                overlaps_BA = bbox_overlaps(p_B, g_A)
                overlaps = torch.max(torch.min(overlaps_AA, overlaps_BB), torch.min(overlaps_AB, overlaps_BA))

            max_overlaps, argmax_overlaps = overlaps.max(dim=1)

            matched_indices = torch.nonzero(max_overlaps >= iou_threshold, as_tuple=False).squeeze()
            unmatched_indices = torch.nonzero(max_overlaps < iou_lower_bound, as_tuple=False).squeeze()

            # sample
            sampled_matched_inds, sampled_unmatched_inds = self.sample_indices(
                    matched_indices, unmatched_indices, num, pos_fraction)

            pred_matched_indices.append(sampled_matched_inds)
            pred_unmatched_indices.append(sampled_unmatched_inds)
            gt_matched_indices.append(argmax_overlaps[sampled_matched_inds])

        return pred_matched_indices, pred_unmatched_indices, gt_matched_indices

    def sample_indices(self, matched_indices, unmatched_indices, N, R):
        # Get the device of the input tensors
        device = matched_indices.device

        # reshape
        matched_indices = matched_indices.view(-1)
        unmatched_indices = unmatched_indices.view(-1)

        # Get the number of positive and negative matches
        num_positive_matches = matched_indices.shape[0]
        num_negative_matches = unmatched_indices.shape[0]

        # Calculate the number of matched indices based on the desired ratio
        num_matched_indices = int(N * R)
        num_matched_indices = min(num_matched_indices, num_positive_matches)

        if num_matched_indices > 0:
            # Sample the required number of matched indices
            sampled_matched_indices = matched_indices[random.sample(range(num_positive_matches), num_matched_indices)]
        else:
            sampled_matched_indices = torch.tensor([], dtype=torch.int64, device=device)

        # Calculate the remaining number of indices to sample
        remaining_indices = N - num_matched_indices

        # Adjust the remaining number of indices if there aren't enough unmatched indices
        remaining_indices = min(remaining_indices, num_negative_matches)

        # Sample the remaining number of unmatched indices
        if remaining_indices > 0:
            sampled_unmatched_indices = unmatched_indices[random.sample(range(num_negative_matches), remaining_indices)]
        else:
            sampled_unmatched_indices = torch.tensor([], dtype=torch.int64, device=device)

        # Return the sampled matched and unmatched indices separately
        return sampled_matched_indices, sampled_unmatched_indices

    def match_boxes_A_only(self, pred_boxes_A, gt_boxes_A, iou_threshold=0.2, iou_lower_bound=0.5, num=50, pos_fraction=0.5):
        """
        Match predicted boxes_A to ground truth boxes_A based on IoU.

        Args:
            pred_boxes_A: List of tensors of shape (N, 4) representing predicted bounding boxes A.
            gt_boxes_A: List of tensors of shape (M, 4) representing ground truth bounding boxes A.
            iou_threshold: IoU threshold for a positive match.
            iou_lower_bound: IoU lower bound to separate unmatched predictions.
            num: Number of samples to retain.
            pos_fraction: Fraction of positive samples in the retained set.

        Returns:
            pred_matched_indices: List of matched indices for predictions.
            pred_unmatched_indices: List of unmatched indices for predictions.
            gt_matched_indices: List of ground truth indices corresponding to matched predictions.
        """
        DEBUG = False  # Toggle debug mode

        B = len(pred_boxes_A)
        
        pred_matched_indices = []
        pred_unmatched_indices = []
        gt_matched_indices = []
        
        for b in range(B):
            p_A = pred_boxes_A[b]  # Predicted boxes_A
            g_A = gt_boxes_A[b]    # Ground truth boxes_A

            # ✅ Debugging: Check tensor shapes
            if DEBUG:
                print(f"Batch {b}: p_A.shape = {p_A.shape}, g_A.shape = {g_A.shape}")

            # Handle edge cases for `g_A`
            if g_A.numel() == 0:  # ✅ If g_A is empty
                if DEBUG:
                    print(f"⚠️ Batch {b}: g_A is empty! Assigning M=0")
                M = 0
                overlaps = torch.zeros(p_A.shape[0], 1, device=p_A.device)
            else:
                if g_A.dim() == 1:  # ✅ If `g_A` is 1D, expand to (M, 4)
                    g_A = g_A.unsqueeze(1)
                    if DEBUG:
                        print(f"🔄 Reshaped g_A to: {g_A.shape}")

                M, _ = g_A.shape  # ✅ Now safe to unpack

            # Handle empty predicted boxes
            if p_A.numel() == 0:  
                pred_matched_indices.append(torch.tensor([], dtype=torch.int64, device=p_A.device))
                pred_unmatched_indices.append(torch.tensor([], dtype=torch.int64, device=p_A.device))
                gt_matched_indices.append(torch.tensor([], dtype=torch.int64, device=p_A.device))
                continue

            # ✅ Compute IoU overlaps
            overlaps_AA = bbox_overlaps(p_A, g_A) if M > 0 else torch.zeros(p_A.shape[0], 1, device=p_A.device)
            overlaps = overlaps_AA  

            # ✅ Get max IoU and corresponding GT index
            max_overlaps, argmax_overlaps = overlaps.max(dim=1)

            # ✅ Filter matched and unmatched indices
            matched_indices = torch.nonzero(max_overlaps >= iou_threshold, as_tuple=False).squeeze()
            unmatched_indices = torch.nonzero(max_overlaps < iou_lower_bound, as_tuple=False).squeeze()

            # ✅ Sample indices
            sampled_matched_inds, sampled_unmatched_inds = self.sample_indices(
                matched_indices, unmatched_indices, num, pos_fraction
            )

            # ✅ Store results
            pred_matched_indices.append(sampled_matched_inds)
            pred_unmatched_indices.append(sampled_unmatched_inds)
            gt_matched_indices.append(argmax_overlaps[sampled_matched_inds])

        return pred_matched_indices, pred_unmatched_indices, gt_matched_indices