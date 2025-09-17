---
layout: project_page
permalink: /

title: "[MICCAI 2025] Towards Holistic Surgical Scene Graph"
authors:
    - Jongmin Shin*¹, Enki Cho*², Ka Young Kim*²,
    - Jung Yong Kim¹, Seong Tae Kim†², Namkee Oh†¹
affiliations:
    - ¹Department of Surgery, Samsung Medical Center, Seoul 06351, Republic of Korea
    - ²Kyung Hee University, Yongin 17104, Republic of Korea
paper: https://arxiv.org/pdf/2507.15541
code: https://github.com/ailab-kyunghee/SSG-Com
---


![Illustration](/static/image/1.png)  
<!-- Abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h5>Abstract</h5>
        <div class="content has-text-justified">
        Surgical scene understanding is crucial for computer-assisted intervention systems, requiring visual comprehension of surgical scenes that involves diverse elements such as surgical tools, anatomical structures, and their interactions. 
        To effectively represent the complex information in surgical scenes, graph-based approaches have been explored to structurally model surgical entities and their relationships. 
        However, aspects such as tool–action–target combinations and the identity of the operating hand remain underexplored. 
        To address this, we propose <b>Endoscapes-SG201</b>, a new dataset including annotations for action triplets(tool–action–target) and hand identity. 
        We also introduce <b>SSG-Com</b>, a graph-based method designed to represent these critical elements. 
        Experiments on downstream tasks—Critical View of Safety (CVS) assessment and action triplet recognition—demonstrate the importance of integrating these scene graph components, significantly advancing holistic surgical scene understanding. 
        </div>
    </div>
</div>

<div class="is-centered">
  <h2>Main Contributions</h2>
  <div class="is-four-fifths has-text-centered">
    <img src="./static/image/2.png" alt="Key Contribution">
  </div>
  <h3>Endoscapes-SG201</h3>
  <img src="./static/image/construction.png">
        <div class="content has-text-justified">

  We were fortunate to build <a href="https://github.com/ailab-kyunghee/SSG-Com">Endoscapes-SG201</a>, a dataset for holistic scene graph research, by extending and refining the publicly available <a href="https://github.com/CAMMA-public/Endoscapes">Endoscapes-Bbox201</a> dataset released by CAMMA.
        </div>
</div>

To annotate additional labels, two clinical experts from Samsung Medical Center refined the bounding boxes in Endoscapes-Bbox201.
- **Step 1**: We refined Bounding Boxes from Endoscapes-Bbox201
- **Step 2**: We subdivided the 'Tool' class into 6 classes
- **Step 3**: We annotated Action labels which describes interactions between tools and anatomical structures, and Hand Identity labels which specifies which hand is manipulating each tool

<div class="is-centered">
    <h4>Dataset Comparison</h4>
    <div class="is-four-fifths has-text-centered">
    <img src="./static/image/3.png" alt="Dataset Comparison">
      </div>
      </div>

This table contrasts the datasets used in previous surgical scene graph studies with Endoscapes-SG201.

- Endoscapes-SG201 is designed with holistic scene graph research in mind.

- It incorporates:
  - Diverse tools and anatomical structures as graph nodes.
  - Diverse relationships as graph edges.
  - Hand Identity labels as attributes of the tool nodes.

- By unifying these elements, the dataset provides a more expressive and comprehensive foundation for modeling surgical scenes.
      
<div class="is-centered">
    <div class="is-four-fifths has-text-centered">
      <div class="content has-text-justified">
      <h4>Endoscapes-SG201 Details</h4>
      <img src="./static/image/4.png" alt="Endoscapes-SG201 Dataset Details">
    </div>
  </div>
</div>

This table presents the category-wise distribution of the additional labels introduced in Endoscapes-SG201.

**Additional Annotations:**
- **6 Surgical Instruments**: Hook (HK), Grasper (GP), Clipper (CL), Bipolar (BP), Irrigator (IG), Scissors (SC)
- **6 Surgical Actions**: Dissect (Dis.), Retract (Ret.), Grasp (Gr.), Clip (Cl.), Coagulate (Co.), Null
- **3 Hand Identities**: Operator’s Right Hand (Rt), Operator’s Left Hand (Lt), Assistant’s Hand (Assi)


<div>
  <div class="is-four-fifths">
    <h3>SSG-Com</h3>
    <img src="./static/image/5.png" alt="SSG-Com Overall Architecture">
  </div>
</div>

SSG-Com is designed to leverage the diverse labels of Endoscapes-SG201.
1. **Graph Construction**  
   - **Nodes**: Surgical instruments(with Hand identity), Anatomical structures  
   - **Edges**: spatial relations, Surgical action relations  
2. **Multi-task training with 3 classifiers**
  - Spatial relation classification
  - Action relation classification
  - Hand identity classification  
  - **Total Loss**:  $L_{\text{total}} = L_{\text{LG}} + \lambda_{\text{action}} L_{\text{action}} + \lambda_{\text{hand}} L_{\text{hand}}$



<div class="is-centered">
  <div class="is-four-fifths">
    <h2>Experimental Results</h2>
  </div>
</div>
      
We employed the latent graph to perform two downstream tasks.
- Action Triplet Recognition
- CVS prediction

<div class="is-centered">
  <div class="is-four-fifths">
    <h3>Quantitative Results</h3>
    <img src="./static/image/6.png" alt="Quantitative Results">
    <div class="content has-text-justified">
      <p>
      </p>
    </div>
    <h3>Qualitative Results</h3>
    <img src="./static/image/7.png" alt="Qualitative Results">
  </div>
</div>

<div class="is-centered">
  <div class="is-four-fifths">
    <img src="./static/image/8.png" alt="Collaborations">
  </div>
</div>
