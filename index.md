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
        <h3>Abstract</h3>
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
  <h2>Main Contribution</h2>
  <div class="is-four-fifths has-text-centered">
    <img src="https://github.com/user-attachments/assets/66a5ba50-36c8-487a-93d4-65ba6601bbeb?download=1" alt="Key Contribution">
  </div>
  <h3>Endoscapes-SG201</h3>
  <div class="is-four-fifths has-text-centered">
    <img src="https://github.com/user-attachments/assets/c8635b4a-522a-43e6-8ab2-f9f430818fa4?download=1" alt="Dataset Construction">
        <div class="content has-text-justified">
        We were fortunate to build Endoscapes-SG201, a dataset for holistic scene graph research, by extending and refining the publicly available Endoscapes-Bbox201 dataset released by CAMMA. Two clinical experts from Samsung Medical Center refined the bounding boxes in Endoscapes-Bbox201, and the original Tool category was subdivided into six distinct classes. In addition, we introduced action labels describing interactions between tools and anatomical structures, as well as Hand Identity labels specifying which hand was manipulating each tool.
        </div>
  </div>
    <h4>Dataset Comparison</h4>
    <div class="is-four-fifths has-text-centered">
    <img src="https://github.com/user-attachments/assets/9b5e9706-b23d-457b-b060-6291af6da80a?download=1" alt="Dataset Comparison">
    <div class="content has-text-justified">
      <p>
        이 테이블은 그동안 써지컬 씬 그래프 논문들에서 사용해온 데이터셋들과 Endoscapes-SG201 사이의 관
      </p>
      </div>
      <div class="content has-text-justified">
      <h4>Endoscapes-SG201 Details</h4>
      <img src="https://github.com/user-attachments/assets/539b67e5-762b-4436-a4fc-847ac6dfc6ab?download=1" alt="Endoscapes-SG201 Dataset Details">
    </div>
  </div>

  <div class="is-four-fifths">
    <h3>SSG-Com</h3>
    <img src="https://github.com/user-attachments/assets/d8214ea0-63c0-43f2-a630-40a33102065d?download=1" alt="SSG-Com Overall Architecture">
    <div class="content has-text-justified">
      <p>
        SSG-Com dataset introduces XXX. 
      </p>
    </div>
  </div>
</div>

<div class="is-centered">
  <div class="is-four-fifths">
    <h2>Experimental Results</h2>
  </div>

  <div class="is-four-fifths">
    <h3>Quantitative Results</h3>
    <img src="https://github.com/user-attachments/assets/5de2d42a-bcfe-4ebb-9cc0-091691dabddb?download=1" alt="Quantitative Results">
    <div class="content has-text-justified">
      <p>
        Endoscapes-SG20 dataset provides XXX.
      </p>
    </div>
    <h3>Qualitative Results</h3>
    <img src="https://github.com/user-attachments/assets/cd1c17d0-3b34-4971-b1b4-3841f567119c?download=1" alt="Qualitative Results">
  </div>
</div>



<div class="is-centered">
  <div class="is-four-fifths">
    <img src="https://github.com/user-attachments/assets/70e6c3a8-672e-4611-907b-717b26b87a49?download=1" alt="Collaborations">
  </div>
</div>


## 🚀 Motivation
Surgical AI systems need a **holistic understanding of surgical scenes**, not just detecting tools and anatomy.  
Previous scene graph approaches overlooked two key aspects:
1. **Tool–Action–Target combinations** (e.g., "grasper–retract–gallbladder")  
2. **Hand identity** (operator’s left/right hand vs. assistant’s hand)

These missing elements are crucial for context-aware modeling in surgical workflows.

---

## 📊 Endoscapes-SG201 Dataset
We construct **Endoscapes-SG201**, built upon Endoscapes-BBox201:
- **1,933 frames** from **201 laparoscopic cholecystectomy videos**
- **Annotations:**
  - 6 surgical tools: Grasper, Hook, Clipper, Bipolar, Irrigator, Scissors
  - 5 anatomical structures
  - 6 surgical actions (Dissect, Retract, Grasp, Clip, Coagulate, Null)
  - 3 hand identities: Operator-Right, Operator-Left, Assistant
- **Splits:** 1,212 train / 409 val / 312 test frames:contentReference[oaicite:2]{index=2}

*Figure 1: Endoscapes-SG201 construction process.*

---

## 🧩 Method: SSG-Com
We propose **SSG-Com (Surgical Scene Graph for Comprehensive Understanding)**:
1. **Graph Construction**  
   - Nodes: tools + anatomical structures  
   - Edges: spatial + surgical action edges  
   - Additional classifier for **hand identity**
2. **Learning**  
   - Multi-task training: spatial relations, action edges, hand identity  
   - Loss: \(L_{total} = L_{LG} + \lambda_{action} L_{action} + \lambda_{hand} L_{hand}\):contentReference[oaicite:3]{index=3}
3. **Downstream Tasks**  
   - **CVS Prediction**  
   - **Action Triplet Recognition**

*Figure 2: SSG-Com framework.*

---

## 📈 Results


*SSG-Com consistently outperforms prior methods, especially when incorporating hand identity.*

---

## 🔍 Qualitative Results
We visualize surgical scene graphs generated by **LG-CVS** and **SSG-Com**:

- **Nodes**: surgical tools & anatomical structures  
- **Edges**:  
  - **Solid straight lines** → spatial relationships (left-right, above-below, inside-outside)  
  - **Dashed curved lines** → action relationships (e.g., grasping, dissecting)  
- **Hand icons** next to tool nodes → indicate operating hand (Rt, Lt, Assistant)

*Figure 3: Comparison of graphs generated by LG-CVS vs. SSG-Com. Our method captures both action relations and hand identity, leading to more contextually rich representations*

---

## ✨ Contributions
- **New dataset**: Endoscapes-SG201 with fine-grained tool, action, and hand annotations.  
- **New method**: SSG-Com, integrating spatial, action, and hand relations.  
- **Improved performance** on both **triplet recognition** and **CVS assessment**, surpassing state-of-the-art baselines.

<!-- --- -->
<!-- 
## 📚 Citation
If you use our work, please cite:

```bibtex
@article{shin2025towards,
  title={Towards Holistic Surgical Scene Graph},
  author={Shin, Jongmin and Cho, Enki and Kim, Ka Young and Kim, Jung Yong and Kim, Seong Tae and Oh, Namkee},
  journal={Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2025}
} -->
