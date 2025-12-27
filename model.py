"""
Main model architecture with CONCH backbone, LoRA, and heads
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from heads import KnowledgeAttentionBlock, MultiScaleSIMHead, ExpertPromptBank, KnowledgeBank


def tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
    """
    Convert tokens to spatial map
    tokens: [B, L, C]
      - If L is square => no CLS
      - If L = 1 + square => has CLS
    return: [B, C, H, W]
    """
    if tokens.dim() != 3:
        raise ValueError(f"Expected [B,L,C], got {tokens.shape}")
    B, L, C = tokens.shape

    s = int(L ** 0.5)
    if s * s == L:
        patch = tokens
    else:
        s = int((L - 1) ** 0.5)
        if s * s != (L - 1):
            raise ValueError(f"L={L} is not square or 1+square.")
        patch = tokens[:, 1:, :]  # drop CLS

    return patch.permute(0, 2, 1).contiguous().view(B, C, s, s)


class LoRALinear(nn.Module):
    """
    LoRA linear layer
    y = base(x) + scaling * B(A(dropout(x)))
    where A: [in -> r], B: [r -> out]
    """
    
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base

        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = (self.alpha / self.r) if self.r > 0 else 0.0
        self.lora_dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

        # freeze base
        for p in self.base.parameters():
            p.requires_grad_(False)

        if self.r > 0:
            self.lora_A = nn.Linear(base.in_features, self.r, bias=False)
            self.lora_B = nn.Linear(self.r, base.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r > 0:
            y = y + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return y


def inject_lora_conch_text(text_tower: nn.Module, r=8, alpha=16, dropout=0.05, mode="attn_mlp"):
    """
    Replace nn.Linear in CONCH text tower by LoRALinear.
    mode:
      - "all"
      - "attn_mlp": only paths contain attn/mlp/proj/fc/qkv...
    """
    mode = str(mode).lower()
    keys = ("attn", "mlp", "c_fc", "c_proj", "qkv", "q_proj", "k_proj", "v_proj", "out_proj", "proj", "fc")

    def iter_with_parent(module, prefix=""):
        for name, child in module.named_children():
            full = f"{prefix}.{name}" if prefix else name
            yield module, name, child, full
            yield from iter_with_parent(child, full)

    replaced = 0
    for parent, name, child, full in iter_with_parent(text_tower):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear):
            if mode == "attn_mlp":
                low = full.lower()
                if not any(k in low for k in keys):
                    continue
            setattr(parent, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            replaced += 1
    return replaced


def conch_lora_sanity_check(conch_model):
    """Sanity check for LoRA injection"""
    t = conch_model.text
    m = t.transformer.resblocks[0].mlp.c_fc
    print("c_fc type:", type(m))
    print("has weight:", hasattr(m, "weight"))
    print("weight dtype:", m.weight.dtype)


def get_lora_parameters(module: nn.Module):
    """Get all LoRA parameters from module"""
    return [p for n, p in module.named_parameters() if ("lora_A" in n or "lora_B" in n) and p.requires_grad]


class ConchPyramidBackbone(nn.Module):
    """CONCH pyramid backbone with persistent hooks"""
    
    def __init__(self, conch_model: nn.Module, block_idxs=(2, 5, 8, 11)):
        super().__init__()
        self.conch = conch_model
        self.block_idxs = tuple(int(i) for i in block_idxs)

        self.vision_trunk = self.conch.visual.trunk
        self.vit_blocks = self.vision_trunk.blocks
        assert isinstance(self.vit_blocks, nn.Sequential), "Expected conch.visual.trunk.blocks as nn.Sequential"

        self._feats = {}
        self._hooks = []
        tags = ["P1", "P2", "P3", "P4"]
        assert len(self.block_idxs) == 4

        def hooker(tag):
            def _hook(_m, _inp, out):
                if isinstance(out, (tuple, list)):
                    out = out[0]
                self._feats[tag] = out
            return _hook

        for tag, bi in zip(tags, self.block_idxs):
            self._hooks.append(self.vit_blocks[bi].register_forward_hook(hooker(tag)))

    def forward(self, images):
        self._feats = {}
        _ = self.conch.encode_image(images, proj_contrast=False, normalize=False)

        out = {}
        for tag in ["P1", "P2", "P3", "P4"]:
            if tag not in self._feats:
                raise RuntimeError(f"Missing {tag} feature from hooks. Check block_idxs.")
            out[tag] = tokens_to_map(self._feats[tag])
        return out


def configure_conch_vision_trainability(conch_model: nn.Module, mode="last_k", last_k=2):
    """Configure CONCH vision trainability"""
    mode = str(mode).lower()
    for p in conch_model.visual.parameters():
        p.requires_grad_(False)

    if mode == "freeze":
        return
    if mode == "full":
        for p in conch_model.visual.parameters():
            p.requires_grad_(True)
        return
    if mode == "last_k":
        trunk = conch_model.visual.trunk
        blocks = trunk.blocks
        K = max(0, min(int(last_k), len(blocks)))
        if K > 0:
            for b in blocks[-K:]:
                for p in b.parameters():
                    p.requires_grad_(True)
        if hasattr(trunk, "norm") and isinstance(trunk.norm, nn.Module):
            for p in trunk.norm.parameters():
                p.requires_grad_(True)
        return

    raise ValueError(f"Unknown CONCH_VISION_TRAIN_MODE: {mode}")


class Model(nn.Module):
    """Joint model with CONCH backbone, prompt bank, and knowledge bank"""
    
    def __init__(self, conch_model, conch_tokenizer, config, prototypes=None, num_classes=4):
        super().__init__()
        self.num_classes = int(num_classes)
        self.config = config

        self.conch = conch_model
        self.tokenizer = conch_tokenizer

        self.vision_dim = int(config["CONCH_VISION_DIM"])   # 768
        self.text_dim = int(config["CONCH_TEXT_DIM"])     # 512

        # 1) Vision backbone (P1..P4)
        self.backbone = ConchPyramidBackbone(self.conch, block_idxs=config["CONCH_BLOCK_IDXS"])

        # 2) Trainability
        configure_conch_vision_trainability(
            self.conch,
            mode=config.get("CONCH_VISION_TRAIN_MODE", "last_k"),
            last_k=config.get("CONCH_VISION_LAST_K", 2),
        )

        # 3) Freeze all conch params then re-enable vision + LoRA
        for p in self.conch.parameters():
            p.requires_grad_(False)

        configure_conch_vision_trainability(
            self.conch,
            mode=config.get("CONCH_VISION_TRAIN_MODE", "last_k"),
            last_k=config.get("CONCH_VISION_LAST_K", 2),
        )
        for n, p in self.conch.text.named_parameters():
            if ("lora_A" in n) or ("lora_B" in n):
                p.requires_grad_(True)

        # 4) PromptBank
        self.prompt_bank = ExpertPromptBank(
            class_names=config["TEXT_CLASS_NAMES"],
            conch_tokenizer=self.tokenizer,
            coarse_templates=config["COARSE_TEMPLATES"],
            fine_template=config["FINE_TEMPLATE"],
            structure_bank=config["STRUCTURE_BANK"],
            color_bank=config["COLOR_BANK"],
            banned_structure_terms=config.get("BANNED_STRUCTURE_TERMS", []),
            max_struct_per_prompt=config["FINE_MAX_STRUCT_PER_PROMPT"],
            max_color_per_prompt=config["FINE_MAX_COLOR_PER_PROMPT"],
            max_fine_per_class=config["FINE_MAX_FINE_PER_CLASS"],
            purify=config["PROMPT_PURIFY"],
            red_thr=config["PURIFY_REDUNDANCY_THR"],
            amb_margin=config["PURIFY_AMBIGUITY_MARGIN"],
            min_keep_per_class=config["PURIFY_MIN_KEEP_PER_CLASS"],
            use_ensemble=config["PROMPT_ENSEMBLE"],
        )

        # 5) KnowledgeBank
        self.knowledge_bank = KnowledgeBank(
            config["BERT_ID"], config["TEXT_CLASS_NAMES"], config["KNOWLEDGE_TEXTS"],
            vision_dim=self.vision_dim,
            tokens_per_class=config["KNOWLEDGE_TOKENS_PER_CLASS"],
            freeze_bert=config["FREEZE_BERT"],
            max_len=config["BERT_MAX_LEN"],
            text_dim=self.text_dim,
        )

        # 6) Knowledge attention on P4 tokens
        self.know_attn1 = KnowledgeAttentionBlock(dim=self.vision_dim, num_heads=8)
        self.know_attn2 = KnowledgeAttentionBlock(dim=self.vision_dim, num_heads=8)

        # 7) SIM head (P2,P3,P4)
        self.sim = MultiScaleSIMHead(
            dims=(self.vision_dim, self.vision_dim, self.vision_dim),
            text_dim=self.text_dim,
            num_classes=self.num_classes
        )

        # 8) Image -> text embedding for fine prompt selection
        self.img_to_text = nn.Sequential(
            nn.Linear(self.vision_dim, self.text_dim),
            nn.LayerNorm(self.text_dim),
        )

        # 9) Class-wise learnable beta
        beta0 = float(config.get("FINE_BETA_INIT", 0.6))
        if config.get("FINE_BETA_LEARNABLE", True):
            self.fine_beta = nn.Parameter(torch.full((self.num_classes,), beta0, dtype=torch.float32))
        else:
            self.register_buffer("fine_beta", torch.full((self.num_classes,), beta0, dtype=torch.float32))

        # 10) Optional image prototypes (FUSE IN TEXT SPACE)
        self.use_img_protos = (prototypes is not None) and bool(config.get("USE_IMAGE_PROTOS", True))
        if self.use_img_protos:
            self.register_buffer("img_prototypes", prototypes)
            proto_dim = int(prototypes.shape[-1])
            self.proto_proj = nn.Linear(proto_dim, self.text_dim) if proto_dim != self.text_dim else nn.Identity()
            self.proto_balance = nn.Parameter(torch.tensor(2.0))
            print(f"Image Prototypes ON: protos={tuple(prototypes.shape)} | proto_dim={proto_dim} -> text_dim={self.text_dim}")

        # misc
        self.know_alpha = float(config.get("KNOWLEDGE_ALPHA_IN_LABEL", 0.7))

        self.fine_topk = int(config.get("FINE_TOPK_PER_CLASS", 6))
        self.fine_temp = float(config.get("FINE_TEMP", 10.0))
        self.fine_margin = float(config.get("FINE_SIM_MARGIN", 0.02))

        self.attr_logit_scale = float(config.get("ATTR_LOGIT_SCALE", 10.0))
        self.attr_margin_weight = float(config.get("ATTR_MARGIN_WEIGHT", 1.0))
        self.attr_margin_sharp = float(config.get("ATTR_MARGIN_SHARP", 10.0))

        self.gate_mode = str(config.get("GATE_MODE", "conf_prob"))
        self.gate_conf_pow = float(config.get("GATE_CONF_POW", 1.0))
        self.gate_prob_tau = float(config.get("GATE_PROB_TAU", 0.2))
        self.gate_prob_sharp = float(config.get("GATE_PROB_SHARPNESS", 8.0))

        # affinity feature projection from P2 (768 -> aff_dim)
        aff_dim = int(config.get("AFF_DIM", 64))
        self.aff_proj = nn.Sequential(
            nn.Conv2d(self.vision_dim, aff_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, aff_dim),
            nn.ReLU(inplace=True),
        )

    @torch.no_grad()
    def build_text_knowledge_bases(self, device):
        """Build prompt bank and knowledge bank"""
        self.prompt_bank.build(self.conch, device=device, enable_cache_tokens=True)
        self.knowledge_bank.build(
            device=device,
            use_web=self.config.get("USE_WEB_KNOWLEDGE", False),
            timeout=self.config.get("WEB_KNOWLEDGE_TIMEOUT", 5.0),
        )

    def _fine_select_topk_mixture(self, img_text, fine_emb, idx_by_class):
        """
        Image-conditioned fine prompt selection
        img_text: [B,512]
        fine_emb: [Nf,512]
        Returns:
          fine_vec:   [B,K,512]
          attr_logits:[B,K]
          attr_conf:  [B,K]
        """
        B, D = img_text.shape
        K = self.num_classes
        device = img_text.device

        fine_vec = img_text.new_zeros((B, K, D))
        attr_logits = img_text.new_zeros((B, K))

        if fine_emb is None or fine_emb.numel() == 0:
            return fine_vec, attr_logits, torch.sigmoid(attr_logits)

        img_text = img_text / (img_text.norm(dim=-1, keepdim=True) + 1e-12)
        fine_emb = fine_emb / (fine_emb.norm(dim=-1, keepdim=True) + 1e-12)

        for c in range(K):
            idxs_cpu = idx_by_class[c]
            if idxs_cpu.numel() == 0:
                continue

            idxs = idxs_cpu.to(device)
            E = fine_emb.index_select(0, idxs)            # [M,D]
            sim = img_text @ E.transpose(0, 1)            # [B,M]

            k = int(min(self.fine_topk, sim.size(1)))
            topv, topi = torch.topk(sim, k=k, dim=1, largest=True, sorted=True)

            E_top = E.index_select(0, topi.reshape(-1)).view(B, k, D)
            w = F.softmax(topv * self.fine_temp, dim=1).unsqueeze(-1)
            vec = (w * E_top).sum(dim=1)
            vec = vec / (vec.norm(dim=-1, keepdim=True) + 1e-12)
            fine_vec[:, c, :] = vec

            # attribute confidence logit: top1 + margin term
            top1 = topv[:, 0]
            margin = (topv[:, 0] - topv[:, 1]) if k >= 2 else (topv[:, 0] * 0.0)
            logit = (top1 * self.attr_logit_scale) + \
                    self.attr_margin_weight * ((margin - self.fine_margin) * self.attr_margin_sharp)
            attr_logits[:, c] = logit

        attr_conf = torch.sigmoid(attr_logits)
        return fine_vec, attr_logits, attr_conf

    def _compute_gate(self, attr_conf, prob_base):
        """Compute gating function"""
        if self.gate_mode == "conf":
            gate = attr_conf
        elif self.gate_mode == "prob":
            gate = torch.sigmoid((prob_base - self.gate_prob_tau) * self.gate_prob_sharp)
        elif self.gate_mode == "conf_prob":
            g1 = attr_conf
            g2 = torch.sigmoid((prob_base - self.gate_prob_tau) * self.gate_prob_sharp)
            gate = g1 * g2
        else:
            gate = attr_conf
        gate = gate.clamp(0.0, 1.0) ** self.gate_conf_pow
        return gate

    def forward_cam_logits_multiscale(self, x):
        """Forward pass returning logits and CAMs"""
        # 1) Vision pyramid
        pyr = self.backbone(x)
        p2, p3, p4 = pyr["P2"], pyr["P3"], pyr["P4"]   # [B,768,h,w]

        # 2) Knowledge attention on P4 tokens
        B, C, H, W = p4.shape
        p4_flat = p4.flatten(2).transpose(1, 2)  # [B,N,768]
        know_tok = self.knowledge_bank.get_knowledge_tokens_vision()  # [K*T,768]
        know_tok_b = know_tok.unsqueeze(0).expand(B, -1, -1)
        p4_flat = self.know_attn1(p4_flat, know_tok_b)
        p4_flat = self.know_attn2(p4_flat, know_tok_b)
        p4 = p4_flat.transpose(1, 2).reshape(B, C, H, W)

        # 3) Text embeddings (WITH grad via CONCH text + LoRA)
        coarse, fine, idx_by_class = self.prompt_bank(self.conch, device=x.device)
        know_text = self.knowledge_bank.get_knowledge_for_labels_textspace()  # [K,512]
        base = coarse + self.know_alpha * know_text
        base = base / (base.norm(dim=-1, keepdim=True) + 1e-12)

        # 4) Base sim -> base logits
        (b2, b3, b4), _ = self.sim(p2, p3, p4, base)
        prob_base = torch.sigmoid((0.5*b2 + 0.75*b3 + 1.0*b4) / 2.25)  # [B,K]

        # 5) Image-conditioned fine selection
        img_pool = F.adaptive_avg_pool2d(p4, 1).flatten(1)  # [B,768]
        img_text = self.img_to_text(img_pool)
        img_text = img_text / (img_text.norm(dim=-1, keepdim=True) + 1e-12)

        fine_vec, attr_logits, attr_conf = self._fine_select_topk_mixture(img_text, fine, idx_by_class)

        # 6) Gate + beta
        gate = self._compute_gate(attr_conf, prob_base)
        beta = torch.clamp(self.fine_beta, 0.0, 2.0).view(1, -1, 1)

        base_b = base.unsqueeze(0).expand(B, -1, -1)
        text_final = base_b + beta * gate.unsqueeze(-1) * fine_vec
        text_final = text_final / (text_final.norm(dim=-1, keepdim=True) + 1e-12)

        # 7) Optional prototype fusion (text space)
        alpha_proto = None
        if self.use_img_protos:
            img_proto = self.proto_proj(self.img_prototypes)  # [P,512]
            img_proto = img_proto / (img_proto.norm(dim=-1, keepdim=True) + 1e-12)

            K = self.num_classes
            P = img_proto.shape[0]
            if P % K == 0:
                ppc = P // K
                img_proto_pooled = img_proto.view(K, ppc, -1).mean(dim=1)
            else:
                img_proto_pooled = torch.stack(img_proto.chunk(K, dim=0)).mean(dim=1)
            img_proto_pooled = img_proto_pooled / (img_proto_pooled.norm(dim=-1, keepdim=True) + 1e-12)

            alpha_proto = torch.sigmoid(self.proto_balance)
            text_final = alpha_proto * text_final + (1.0 - alpha_proto) * img_proto_pooled.unsqueeze(0)
            text_final = text_final / (text_final.norm(dim=-1, keepdim=True) + 1e-12)

        # 8) Final multi-scale SIM
        (l2, l3, l4), (cam2, cam3, cam4) = self.sim(p2, p3, p4, text_final)

        # 9) Fuse CAMs at cam2 resolution
        target_hw = cam2.shape[-2:]
        cam3_up = F.interpolate(cam3, size=target_hw, mode="bilinear", align_corners=False)
        cam4_up = F.interpolate(cam4, size=target_hw, mode="bilinear", align_corners=False)
        fused_cam = (0.2 * cam2 + 0.5 * cam3_up + 1.3 * cam4_up) / 2.0

        # 10) Fuse logits
        fused_logits = (0.5 * l2 + 0.75 * l3 + 1.0 * l4) / 2.25

        # 11) Feature for affinity pseudo
        feat_aff = self.aff_proj(p2)

        extras = {
            "attr_logits": attr_logits,
            "attr_conf": attr_conf.detach(),
            "gate": gate.detach(),
            "prob_base": prob_base.detach(),
            "has_fine": bool(fine is not None and fine.numel() > 0),
            "alpha_proto": (alpha_proto.detach().item() if alpha_proto is not None else None),
            "feat_aff": feat_aff.detach(),
        }
        return fused_logits, fused_cam, (l2, l3, l4), (cam2, cam3, cam4), extras

