# Model Architecture Flow

Tài liệu này mô tả chi tiết luồng xử lý của model BCSS-WSSS với CONCH backbone.

## Tổng quan

Model sử dụng CONCH (Contrastive learning for histopathology) làm backbone, kết hợp với Expert Prompt Bank, Knowledge Bank (BERT), và Multi-scale Similarity Head.

---

## 🔹 FORWARD PASS FLOW

```
┌─────────────────────────────────────────────────────────────┐
│  PROTOTYPE GENERATION (Pre-training, Optional)              │
│                                                              │
│  Input:                                                      │
│  • Training images (only single-label, 1 class per image)   │
│  • Label CSV                                                 │
│                                                              │
│  Process:                                                    │
│  1) Filter images:                                          │
│     • Keep only images with exactly 1 class                 │
│     • Skip multi-label images                                │
│                                                              │
│  2) Extract features:                                        │
│     • Load CONCH vision encoder                              │
│     • For each image:                                        │
│       - Resize to 448×448                                    │
│       - Extract features via encode_image()                  │
│       - Normalize                                            │
│     • Group by class                                         │
│                                                              │
│  3) K-means clustering:                                     │
│     • For each class:                                        │
│       - Cosine similarity K-means                           │
│       - k clusters per class (from k_list)                  │
│       - Get cluster centers                                  │
│                                                              │
│  4) Save prototypes:                                         │
│     • Concatenate all cluster centers                       │
│     • Shape: [P, D_proto] where P = sum(k_list)              │
│     • Save to .pkl file                                      │
│                                                              │
│  Output:                                                     │
│  • Image Prototypes [P, D_proto]                            │
│    (Loaded into model as buffer)                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            │  (Prototypes loaded once)
                            │
                            ▼
[Input Image] I [B, 3, 448, 448]
        │
        ▼
 ┌─────────────────────┐
 │  CONCH Vision        │   (ViT-B/16)
 │  Backbone            │
 │                      │
 │  Forward Pass:       │
 │  • Hook tại block 2  │ → P1 [B,768,H1,W1] --> không dùng 
 │  • Hook tại block 5  │ → P2 [B,768,H2,W2] 
 │  • Hook tại block 8  │ → P3 [B,768,H3,W3] 
 │  • Hook tại block 11 │ → P4 [B,768,H4,W4] 
 │                      │
 │  • Train: last_k=2   │
 │  • Frozen: others    │
 └─────────────────────┘
        │
        │  Tất cả được extract song song từ hooks
        │
        ├──────────────┬──────────────┬──────────────┐
        │              │              │              │
        ▼              ▼              ▼              ▼
 ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
 │  P2 Features │ │  P3 Features │ │  P4 Features │
 │  [B,768,H2,W2]│ │  [B,768,H3,W3]│ │  [B,768,H4,W4]│
 │  (block 5)   │ │  (block 8)   │ │  (block 11)  │
 └──────────────┘ └──────────────┘ └──────────────┘
        │              │              │
        │              │              │  Flatten to Tokens
        │              │              ▼
        │              │      ┌──────────────────────┐
        │              │      │  P4 Tokens            │
        │              │      │  [B, N, 768]          │
        │              │      └──────────────────────┘
        │              │              │
        │              │              │
        │              │              │  (chạy song song)
        │              │              │
        ├──────────────┴──────────────┴───────────────────┐
        │                                                   │
        ▼                                                   ▼
┌──────────────────────┐                    ┌──────────────────────┐
│  Knowledge Bank      │                    │  Expert Prompt Bank  │
│  (BERT)              │                    │                      │
│                      │                    │  • Coarse Prompts   │
│  Input:              │                    │    (WITH grad)       │
│  • Class names       │                    │  • Fine Prompts     │
│  • Knowledge texts   │                    │    (NO grad, cached) │
│  (không phụ thuộc P4)│                    │  (không phụ thuộc P4)│
│                      │                    │                      │
│  Process:            │                    │  Process:           │
│  • BERT encode       │                    │  • Generate prompts  │
│  • CLS token         │                    │  • Purification      │
│    [K, bert_dim]     │                    │  • Tokenize          │
│  • Project to:       │                    │  • Encode (LoRA)     │
│    - proj_to_vision: │                    │                      │
│      Linear(bert_dim│                    │  Output:            │
│      → 768)          │                    │  • Coarse [K,512]    │
│      → [K,768]       │                    │  • Fine [Nf,512]     │
│    - proj_to_text:   │                    │  • idx_by_class      │
│      Linear(bert_dim│                    │                      │
│      → 512)           │                    │                      │
│      → [K,512]       │                    │                      │
│                      │                    │                      │
│  Output:             │                    │                      │
│  • Know_tok [K,768]  │                    │                      │
│  • Know_text [K,512] │                    │                      │
└──────────────────────┘                    └──────────────────────┘
        │                                       │
        │  (Knowledge Tokens)                    │  (Text Embeddings)
        │                                       │
        │                                       │
        ├───────────────────────────────────────┤
        │                                       │
        │                                       │
        ▼                                       │
┌──────────────────────────────────────────────────────────┐
│  Knowledge Attention (2 layers)                        │
│                                                          │
│  Input:                                                  │
│  • P4 Tokens [B, N, 768]  ←───────────────────────────┐ │
│  • Knowledge Tokens [K, 768] → expand to [B, K, 768] ──┘ │
│                                                          │
│  Layer 1:                                                │
│  • Concat: [B, N+K, 768]                                 │
│  • LayerNorm1                                            │
│  • Multi-head Self-Attention                             │
│  • Residual: x = x + attn_out                           │
│  • LayerNorm2                                             │
│  • MLP:                                                  │
│    - Linear(768 → 3072) [hidden = dim×4]                │
│    - GELU                                                │
│    - Dropout                                             │
│    - Linear(3072 → 768)                                  │
│    - Dropout                                              │
│  • Residual: x = x + mlp_out                             │
│                                                          │
│  Layer 2:                                                │
│  • Same structure as Layer 1                             │
│                                                          │
│  • Extract image tokens: [B, N, 768]                     │
│                                                          │
│  Output:                                                 │
│  • Enhanced P4 [B, 768, H4, W4]                         │
└──────────────────────────────────────────────────────────┘
                            │
                            │  Enhanced P4
                            │
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        │  (Text embeddings từ Prompt Bank      │  (Vision features)
        │   và Knowledge Bank)                  │
        │                                       │
        ▼                                       ▼
┌──────────────────────┐            ┌──────────────────────┐
│  Base Text Fusion    │            │  Base Similarity      │
│                      │            │  Computation         │
│  Input:              │            │                      │
│  • Coarse [K,512]    │            │  Input:               │
│    (từ Prompt Bank)  │            │  • Base [K,512]        │
│  • Know_text [K,512] │            │    (từ Text Fusion)   │
│    (từ Knowledge Bank)│            │  • P2, P3, Enhanced P4│
│                      │            │                      │
│  Process:            │            │  Process:             │
│  • base = coarse +   │            │  For each scale:      │
│    α × know_text     │            │  • to_w2: Linear(512→768)│
│  • Normalize         │            │    + LayerNorm        │
│                      │            │  • to_w3: Linear(512→768)│
│  Output:             │            │    + LayerNorm        │
│  • Base [K,512] ──────┼──────────→│  • to_w4: Linear(512→768)│
│                      │            │    + LayerNorm        │
│                      │            │  • Normalize text & vision│
│                      │            │  • Compute similarity  │
│                      │            │  • Apply logit_scale  │
│                      │            │  • GAP → logits        │
│                      │            │                      │
│                      │            │  Output:              │
│                      │            │  • l2, l3, l4 [B,K]  │
│                      │            │  • prob_base [B,K]   │
└──────────────────────┘            └──────────────────────┘
        │                                       │
        └───────────────────┬───────────────────┘
                            │
                            ▼
                    ┌──────────────────────┐
                    │  Image-conditioned   │
                    │  Fine Selection      │
                    │                      │
                    │  Input:              │
                    │  • P4 Enhanced       │
                    │  • Fine Cache [Nf,512]│
                    │                      │
                    │  Process:            │
                    │  • Pool P4 → [B,768] │
                    │  • img_to_text:      │
                    │    Linear(768→512)   │
                    │    + LayerNorm       │
                    │    → [B,512]         │
                    │  • Normalize         │
                    │  • For each class:    │
                    │    - Select fine     │
                    │      embeddings      │
                    │      from idx_by_class│
                    │    - Compute sim     │
                    │      [B, M]          │
                    │    - Top-K (k=6)     │
                    │    - Weighted mix    │
                    │      (temp=10.0)     │
                    │    - Attr logit:     │
                    │      top1×scale +     │
                    │      margin×sharp    │
                    │                      │
                    │  Output:             │
                    │  • Fine_vec [B,K,512]│
                    │  • Attr_logits [B,K] │
                    │  • Attr_conf [B,K]   │
                    └──────────────────────┘
                            │
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
┌──────────────────────┐            ┌──────────────────────┐
│  Gating Function     │            │  Text Final Fusion    │
│                      │            │                      │
│  Input:              │            │  Input:              │
│  • Attr_conf [B,K]   │            │  • Base [K,512]      │
│  • prob_base [B,K]   │            │  • Fine_vec [B,K,512] │
│                      │            │  • Gate [B,K]        │
│  Mode: conf_prob     │            │  • Beta [K]          │
│  • g1 = attr_conf    │            │                      │
│  • g2 = sigmoid(     │            │  Process:             │
│      (prob-τ)×sharp) │            │  • text_final = base +│
│  • gate = g1 × g2    │            │    β × gate × fine   │
│  • gate = gate^pow    │            │  • Normalize         │
│                      │            │                      │
│  Output:             │            │  Output:             │
│  • Gate [B,K]       │            │  • Text_final        │
│                      │            │    [B,K,512]         │
└──────────────────────┘            └──────────────────────┘
                            │
                            │
                            │
                            │  Text_final [B,K,512]
                            │
                            ▼
                    ┌──────────────────────┐
                    │  Prototype Fusion    │
                    │  (Optional)          │
                    │                      │
                    │  Input:              │
                    │  • Text_final        │
                    │    [B,K,512]         │
                    │  • Image Prototypes  │
                    │    [P, D_proto]      │
                    │                      │
                    │  Process:            │
                    │  • proto_proj:       │
                    │    Linear(D_proto→512)│
                    │    → [P, 512]        │
                    │  • Normalize         │
                    │  • Pool per class:   │
                    │    mean([P/K, 512])  │
                    │    → [K, 512]        │
                    │  • α_proto = sigmoid │
                    │    (proto_balance)   │
                    │  • text_final =      │
                    │    α_proto × text +  │
                    │    (1-α_proto) ×     │
                    │    proto_pooled      │
                    │  • Normalize         │
                    │                      │
                    │  Output:             │
                    │  • Text_final        │
                    │    [B,K,512]         │
                    └──────────────────────┘
                            │
                            │
                            ▼
                    ┌──────────────────────┐
                    │  Final Multi-scale  │
                    │  Similarity         │
                    │                      │
                    │  Input:             │
                    │  • Text_final        │
                    │    [B,K,512]         │
                    │  • P2, P3, P4        │
                    │                      │
                    │  Process:            │
                    │  • Multi-scale SIM  │
                    │  • Similarity maps   │
                    │  • GAP → logits     │
                    │                      │
                    │  Output:             │
                    │  • l2, l3, l4 [B,K] │
                    │  • cam2, cam3, cam4  │
                    │    [B,K,H,W]        │
                    └──────────────────────┘
                            │
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
┌──────────────────────┐            ┌──────────────────────┐
│  CAM Fusion          │            │  Logit Fusion         │
│                      │            │                      │
│  Input:              │            │  Input:              │
│  • cam2, cam3, cam4  │            │  • l2, l3, l4 [B,K]  │
│                      │            │                      │
│  Process:            │            │  Process:             │
│  • Upsample cam3,4   │            │  • Weighted average   │
│    to cam2 size      │            │  • w2=0.5, w3=0.75,  │
│  • Fused = (0.2×cam2 │            │    w4=1.0            │
│    + 0.5×cam3_up     │            │                      │
│    + 1.3×cam4_up)/2.0│            │  Output:             │
│                      │            │  • Fused_logits      │
│  Output:             │            │    [B,K]            │
│  • Fused_cam         │            │                      │
│    [B,K,H2,W2]       │            │                      │
└──────────────────────┘            └──────────────────────┘
                            │
                            │
                            ▼
                    ┌──────────────────────┐
                    │  Affinity Features   │
                    │                      │
                    │  Input:              │
                    │  • P2 [B,768,H2,W2]  │
                    │                      │
                    │  Process:            │
                    │  • aff_proj:         │
                    │    Conv2d(768→64,    │
                    │     kernel=1×1)      │
                    │    + GroupNorm(8)    │
                    │    + ReLU            │
                    │                      │
                    │  Output:             │
                    │  • Aff_feat          │
                    │    [B,64,H2,W2]      │
                    └──────────────────────┘
                            │
                            │
                            ▼
                    ┌──────────────────────┐
                    │  OUTPUT              │
                    │                      │
                    │  • Fused_logits      │
                    │    [B, K]            │
                    │  • Fused_cam         │
                    │    [B, K, H, W]      │
                    │  • Multi-scale       │
                    │    logits & CAMs     │
                    │  • Affinity features │
                    │  • Extras:           │
                    │    - attr_logits     │
                    │    - attr_conf       │
                    │    - gate            │
                    │    - prob_base       │
                    │    - has_fine        │
                    │    - alpha_proto     │
                    │    - feat_aff        │
                    └──────────────────────┘
```

---

## 🔹 COMPONENT DETAILS

### 1. CONCH Vision Backbone

```
Input: [B, 3, 448, 448]
    ↓
CONCH Visual Encoder (ViT-B/16)
    ↓
Forward Hooks tại blocks [2, 5, 8, 11]
    ↓
P1 [B, 768, H1, W1]  (block 2)  - không dùng
P2 [B, 768, H2, W2]  (block 5)  ← Sử dụng
P3 [B, 768, H3, W3]  (block 8)  ← Sử dụng
P4 [B, 768, H4, W4]  (block 11) ← Sử dụng
```

**Trainability**: Chỉ train last_k blocks (mặc định: 2 blocks cuối)

### 2. Knowledge Bank (BERT)

```
┌──────────────────────────────────────┐
│  Knowledge Bank                      │
│                                      │
│  Input:                              │
│  • Class names [K]                   │
│  • Knowledge texts (dict)            │
│                                      │
│  Process:                            │
│  • BERT Tokenizer                    │
│  • BERT Encoder (frozen)             │
│  • Extract CLS token [K, bert_dim]   │
│  • Projection layers:                │
│    - proj_to_vision:                 │
│      Linear(bert_dim → 768)          │
│      → [K, 768]                      │
│    - proj_to_text:                   │
│      Linear(bert_dim → 512)          │
│      → [K, 512]                      │
│                                      │
│  Output:                             │
│  • know_vec_bert [K, bert_dim]       │
│  • know_tok [K, 768] (vision)        │
│  • know_text [K, 512] (text)         │
└──────────────────────────────────────┘
```

**Trainability**: BERT frozen, chỉ train projection layers (proj_to_vision, proj_to_text)

### 3. Expert Prompt Bank

```
┌──────────────────────────────────────┐
│  Expert Prompt Bank                  │
│                                      │
│  ┌────────────────────────────────┐ │
│  │ Coarse Prompts (WITH grad)     │ │
│  │                                │ │
│  │ • Class × Templates            │ │
│  │ • Tokenize                      │ │
│  │ • CONCH encode (LoRA)          │ │
│  │ • Average per class             │ │
│  │                                │ │
│  │ Output: [K, 512]               │ │
│  └────────────────────────────────┘ │
│                                      │
│  ┌────────────────────────────────┐ │
│  │ Fine Prompts (NO grad, cached) │ │
│  │                                │ │
│  │ • Generate combinations:       │ │
│  │   - Structure × Color          │ │
│  │   - Structure only             │ │
│  │ • Purification:                │ │
│  │   - Ambiguity filter           │ │
│  │   - Redundancy removal         │ │
│  │ • Tokenize                      │ │
│  │ • CONCH encode (no_grad)       │ │
│  │ • Cache [Nf, 512]              │ │
│  │                                │ │
│  │ Refresh: mỗi epoch              │ │
│  └────────────────────────────────┘ │
└──────────────────────────────────────┘
```

### 4. Image-conditioned Fine Selection

```
┌──────────────────────────────────────┐
│  Fine Prompt Selection               │
│                                      │
│  Input:                              │
│  • P4 Enhanced [B,768,H4,W4]         │
│  • Fine cache [Nf, 512]              │
│  • idx_by_class (list)               │
│                                      │
│  Process:                            │
│  • Pool P4 → [B, 768]                │
│  • img_to_text projection:           │
│    Linear(768 → 512)                 │
│    + LayerNorm                       │
│    → [B, 512]                        │
│  • Normalize                         │
│                                      │
│  For each class c:                   │
│    • Select fine embeddings [M,512]  │
│    • Compute similarity [B, M]        │
│    • Top-K selection (k=6)           │
│    • Weighted mixture:                │
│      w = softmax(sim × temp=10.0)    │
│      vec = Σ w_i × emb_i             │
│    • Attribute confidence:            │
│      conf = sigmoid(top1 + margin)   │
│                                      │
│  Output:                             │
│  • Fine_vec [B, K, 512]              │
│  • Attr_logits [B, K]                │
│  • Attr_conf [B, K]                  │
└──────────────────────────────────────┘
```

### 5. Confidence-gated Fusion

```
┌──────────────────────────────────────┐
│  Gating Mechanism                    │
│                                      │
│  Mode: conf_prob                     │
│                                      │
│  • g1 = attr_conf [B, K]            │
│  • g2 = sigmoid((prob_base - τ)     │
│              × sharpness)            │
│  • gate = g1 × g2                   │
│  • gate = gate^pow                  │
│                                      │
│  Text Final:                         │
│  • base [B, K, 512]                  │
│  • + β [K] × gate [B,K,1]            │
│      × fine [B,K,512]                │
│  • Normalize                         │
│                                      │
│  Output:                             │
│  • Text_final [B, K, 512]            │
└──────────────────────────────────────┘
```

### 6. Multi-scale Similarity Head

```
┌──────────────────────────────────────┐
│  Multi-scale SIM Head                 │
│                                      │
│  Input:                              │
│  • Text [K,512] or [B,K,512]        │
│  • Vision features P2, P3, P4        │
│                                      │
│  Projection layers:                  │
│  • to_w2: Linear(512→768) + LayerNorm│
│  • to_w3: Linear(512→768) + LayerNorm│
│  • to_w4: Linear(512→768) + LayerNorm│
│                                      │
│  For each scale (P2, P3, P4):        │
│    • Project text → vision dim       │
│      using corresponding to_w        │
│    • Normalize both                  │
│    • Compute similarity:             │
│      sim = einsum("bchw,kc->bkhw")   │
│    • Apply logit_scale (learnable)   │
│    • GAP → logits [B, K]            │
│                                      │
│  Output:                             │
│  • Logits: l2, l3, l4 [B, K]        │
│  • CAMs: cam2, cam3, cam4            │
│      [B, K, H, W]                   │
└──────────────────────────────────────┘
```

---

## 🔹 TRAINING FLOW

```
[Training Batch]
        │
        ▼
┌──────────────────────┐
│  Forward Pass        │
│  • Get logits & CAMs │
└──────────────────────┘
        │
        ├───────────────────────────────────────────────┐
        │                                               │
        ▼                                               ▼
┌──────────────────────┐                    ┌──────────────────────┐
│  Multi-scale Loss    │                    │  Equivariance Loss  │
│                      │                    │                      │
│  L_ms = (w2×BCE(l2) +│                    │  L_eq = ||CAM(img) - │
│         w3×BCE(l3) + │                    │         flip(CAM(    │
│         w4×BCE(l4)) /│                    │         flip(img)))|| │
│         (w2+w3+w4)   │                    │                      │
└──────────────────────┘                    └──────────────────────┘
        │                                               │
        │                                               │
        └───────────────────┬───────────────────────────┘
                            │
                            ▼
                    ┌──────────────────────┐
                    │  Attribute Loss      │
                    │  (Optional)          │
                    │                      │
                    │  L_attr = BCE(        │
                    │    attr_logits,      │
                    │    labels)           │
                    └──────────────────────┘
                            │
                            ▼
                    ┌──────────────────────┐
                    │  Total Loss         │
                    │                      │
                    │  L = L_ms +          │
                    │      λ×L_eq +        │
                    │      w_attr×L_attr   │
                    │                      │
                    │  Where:             │
                    │  • λ = 0.15×min(1,  │
                    │        epoch/3)      │
                    │  • w_attr = 0.05     │
                    └──────────────────────┘
                            │
                            ▼
                    ┌──────────────────────┐
                    │  Backward & Update   │
                    │                      │
                    │  • Vision params     │
                    │    (last_k blocks)   │
                    │  • LoRA params      │
                    │  • Head params      │
                    │                      │
                    │  Optimizers:         │
                    │  • Adam (vision)    │
                    │  • Adam (head)      │
                    │  • Adam (lora)      │
                    └──────────────────────┘
```

---

## 🔹 EVALUATION FLOW

```
[Validation/Test Image]
        │
        ▼
┌──────────────────────┐
│  Forward Pass        │
│  (with TTA)          │
│                      │
│  • Multiple scales   │
│  • Horizontal flip   │
│  • Average results   │
└──────────────────────┘
        │
        ├───────────────────────────────────────────────┐
        │                                               │
        ▼                                               ▼
┌──────────────────────┐                    ┌──────────────────────┐
│  Classification Eval  │                    │  Pseudo mIoU Eval     │
│                      │                    │                      │
│  • Compute metrics:  │                    │  • CAM → Mask        │
│    - F1 (micro/macro)│                    │  • Affinity prop      │
│    - AUC-ROC         │                    │  • Tissue mask        │
│    - mAP             │                    │  • Compute mIoU       │
│  • Tune thresholds   │                    │                      │
└──────────────────────┘                    └──────────────────────┘
```

---

## 🔹 DIMENSIONALITY SUMMARY

| Component | Shape | Description |
|-----------|-------|-------------|
| Input Image | [B, 3, 448, 448] | RGB image |
| P2 | [B, 768, H2, W2] | Vision feature level 2 (block 5) |
| P3 | [B, 768, H3, W3] | Vision feature level 3 (block 8) |
| P4 | [B, 768, H4, W4] | Vision feature level 4 (block 11) |
| Knowledge Tokens | [K, 768] | BERT knowledge in vision space |
| Knowledge Text | [K, 512] | BERT knowledge in text space |
| Coarse Embeddings | [K, 512] | Coarse prompts per class |
| Fine Embeddings | [Nf, 512] | Cached fine prompts |
| Base Text | [K, 512] | Coarse + α×Knowledge |
| Image Text | [B, 512] | Image-conditioned embedding |
| Fine Vectors | [B, K, 512] | Selected fine prompts |
| Text Final | [B, K, 512] | Final text embedding (after gate + beta + optional prototype) |
| Image Prototypes | [P, D_proto] | Prototype features (P = sum of k_list per class) |
| Prototype Pooled | [K, 512] | Per-class mean of prototypes |
| Logits | [B, K] | Classification logits (K=4) |
| CAMs | [B, K, H, W] | Class activation maps |
| Affinity Features | [B, 64, H2, W2] | For pseudo mask generation |

---

## 🔹 KEY DESIGN CHOICES

### 1. Fine Prompt Caching
- Fine prompts được encode một lần (no_grad) và cache
- Refresh mỗi epoch để track LoRA updates
- Tránh OOM khi training

### 2. Image-conditioned Selection
- Fine prompts được chọn dựa trên image content
- Top-K selection với temperature-weighted mixture
- Attribute confidence từ margin term

### 3. Confidence Gating
- Kết hợp attribute confidence và base probability
- Điều khiển contribution của fine prompts
- Mode: conf_prob (g1 × g2)

### 4. Multi-scale Fusion
- Sử dụng 3 scales (P2, P3, P4)
- Weights: 0.5, 0.75, 1.0 cho logits
- Weights: 0.2, 0.5, 1.3 cho CAMs

### 5. Knowledge Integration
- BERT knowledge vào cả vision space (attention với MLP)
- Và text space (fusion với α=0.7)
- Knowledge Attention Block: 2 layers, mỗi layer có MLP (768→3072→768)

### 6. LoRA for Text
- Chỉ train LoRA parameters trong CONCH text tower
- Efficient fine-tuning của text encoder

### 7. Image Prototypes (Optional)
- Prototypes được generate từ ảnh training có đúng 1 class (single-label)
- Sử dụng CONCH vision encoder để extract features
- K-means clustering per class để tạo cluster centers
- Prototypes được project từ vision space sang text space qua `proto_proj`: Linear(D_proto→512)
- Fusion với text_final: `α_proto × text + (1-α_proto) × proto`
- α_proto là learnable parameter (sigmoid(proto_balance))

### 8. Projection Layers Summary
- **Knowledge Bank**: proj_to_vision (bert_dim→768), proj_to_text (bert_dim→512)
- **Knowledge Attention**: MLP trong mỗi layer (768→3072→768 với GELU)
- **Multi-scale SIM**: to_w2, to_w3, to_w4 (512→768 + LayerNorm)
- **Image-conditioned Selection**: img_to_text (768→512 + LayerNorm)
- **Prototype Fusion**: proto_proj (D_proto→512)
- **Affinity Features**: aff_proj (Conv2d 768→64 + GroupNorm + ReLU)

---

## 🔹 CLASS NAMES

- **Tumor** (Class 0) - "tumor epithelium"
- **Stroma** (Class 1) - "tumor-associated stroma"
- **Lymphocytic infiltrate** (Class 2) - "lymphocyte infiltrate"
- **Necrosis** (Class 3) - "necrosis"

---

## 🔹 OUTPUT FORMAT

Model trả về trong `forward_cam_logits_multiscale()`:

```python
return (
    fused_logits,      # [B, K] - Classification logits
    fused_cam,         # [B, K, H, W] - Fused CAMs
    (l2, l3, l4),      # Multi-scale logits
    (cam2, cam3, cam4), # Multi-scale CAMs
    extras             # Dictionary:
                       #   - attr_logits [B, K]
                       #   - attr_conf [B, K]
                       #   - gate [B, K]
                       #   - prob_base [B, K]
                       #   - has_fine (bool)
                       #   - alpha_proto (float or None)
                       #   - feat_aff [B, 64, H, W]
)
```
