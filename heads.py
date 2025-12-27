"""
Heads, PromptBank, and KnowledgeBank modules
"""
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm.auto import tqdm

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    raise ImportError("transformers library is required. Install with: pip install transformers")


class KnowledgeAttentionBlock(nn.Module):
    """Knowledge attention block for fusing BERT knowledge with vision features"""
    
    def __init__(self, dim=768, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, img_tokens, know_tokens):
        x = torch.cat([img_tokens, know_tokens], dim=1)  # [B,N+M,C]
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x[:, :img_tokens.size(1), :]


class MultiScaleSIMHead(nn.Module):
    """
    Multi-scale similarity head for CONCH pyramid features
    CONCH pyramid: P2,P3,P4 all [B,768,H,W]
    text_feat: [K,512] or [B,K,512]
    """
    
    def __init__(self, dims=(768, 768, 768), text_dim=512, num_classes=4, use_logit_scale=True):
        super().__init__()
        d2, d3, d4 = dims
        self.to_w2 = nn.Sequential(nn.Linear(text_dim, d2), nn.LayerNorm(d2))
        self.to_w3 = nn.Sequential(nn.Linear(text_dim, d3), nn.LayerNorm(d3))
        self.to_w4 = nn.Sequential(nn.Linear(text_dim, d4), nn.LayerNorm(d4))
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.use_logit_scale = bool(use_logit_scale)
        if self.use_logit_scale:
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(10.0)))  # exp~10

    def _project_norm(self, to_w, text_feat):
        W = to_w(text_feat)
        return F.normalize(W, dim=-1)

    def _sim(self, patch_2d, text_feat, to_w):
        patch = F.normalize(patch_2d, dim=1)  # [B,C,H,W]

        if text_feat.dim() == 2:
            W = self._project_norm(to_w, text_feat)      # [K,C]
            cam = torch.einsum("bchw,kc->bkhw", patch, W)
        elif text_feat.dim() == 3:
            W = self._project_norm(to_w, text_feat)      # [B,K,C]
            cam = torch.einsum("bchw,bkc->bkhw", patch, W)
        else:
            raise ValueError(f"text_feat must be [K,D] or [B,K,D], got {text_feat.shape}")

        if self.use_logit_scale:
            cam = cam * self.logit_scale.exp().clamp(1.0, 100.0)

        logits = self.gap(cam).flatten(1)  # [B,K]
        return logits, cam

    def forward(self, f2, f3, f4, text_feat):
        l2, cam2 = self._sim(f2, text_feat, self.to_w2)
        l3, cam3 = self._sim(f3, text_feat, self.to_w3)
        l4, cam4 = self._sim(f4, text_feat, self.to_w4)
        return (l2, l3, l4), (cam2, cam3, cam4)


class ExpertPromptBank(nn.Module):
    """
    Expert Prompt Bank for CONCH
    - Cache token IDs tensors
    - Purification uses no_grad encode_text (chunked)
    - Forward returns:
        coarse_by_class: WITH grad (train LoRA)
        fine_emb: NO grad (cached) to avoid OOM
    """
    
    def __init__(
        self,
        class_names,
        conch_tokenizer,
        coarse_templates,
        fine_template,
        structure_bank,
        color_bank,
        banned_structure_terms=None,
        max_struct_per_prompt=2,
        max_color_per_prompt=1,
        max_fine_per_class=80,
        purify=True,
        red_thr=0.92,
        amb_margin=0.05,
        min_keep_per_class=25,
        max_len=64,
        use_ensemble=True,
    ):
        super().__init__()
        self.class_names = list(class_names)
        self.tokenizer = conch_tokenizer
        self.coarse_templates = list(coarse_templates)
        self.fine_template = str(fine_template)

        self.structure_bank = {k: list(v) for k, v in structure_bank.items()}
        self.color_bank = {k: list(v) for k, v in color_bank.items()}
        self.banned_structure_terms = [t.lower() for t in (banned_structure_terms or [])]

        self.max_struct_per_prompt = int(max_struct_per_prompt)
        self.max_color_per_prompt = int(max_color_per_prompt)
        self.max_fine_per_class = int(max_fine_per_class)

        self.purify = bool(purify)
        self.red_thr = float(red_thr)
        self.amb_margin = float(amb_margin)
        self.min_keep_per_class = int(min_keep_per_class)

        self.max_len = int(max_len)
        self.use_ensemble = bool(use_ensemble)

        self._ready = False
        self._context_length = 77

        self._coarse_texts = None
        self._fine_texts = None
        self._fine_class_ids = None
        self._idx_by_class_cpu = None

        self._tok_coarse_ids = None
        self._tok_fine_ids = None

        # fine cache (NO grad)
        self._fine_emb_cache = None
        self._fine_cache_meta = {"dtype": None, "device": None, "n": 0}

    def _is_banned_structure(self, s: str) -> bool:
        ss = s.lower().strip()
        return any(t in ss for t in self.banned_structure_terms)

    def _generate_fine_prompts_for_class(self, cname: str):
        structures = [s for s in self.structure_bank.get(cname, []) if not self._is_banned_structure(s)]
        colors = list(self.color_bank.get(cname, []))

        fine_prompts = []

        struct_sets = []
        for r in range(1, self.max_struct_per_prompt + 1):
            for comb in itertools.combinations(structures, r):
                struct_sets.append(comb)

        color_sets = []
        for r in range(1, self.max_color_per_prompt + 1):
            for comb in itertools.combinations(colors, r):
                color_sets.append(comb)

        for ss in struct_sets:
            s_text = ", ".join(ss)
            for cc in color_sets:
                p_text = ", ".join(cc)
                fine_prompts.append(self.fine_template.format(y=cname, s=s_text, p=p_text))

        for ss in struct_sets:
            s_text = ", ".join(ss)
            fine_prompts.append(f"a photo of {cname} tissue, with {s_text} structures.")

        fine_prompts = list(dict.fromkeys([t.strip() for t in fine_prompts if len(t.strip()) > 0]))
        
        # Fallback: if no prompts generated, create at least one basic prompt
        if len(fine_prompts) == 0:
            if len(structures) > 0 and len(colors) > 0:
                # Use first structure and color if available
                s_text = structures[0]
                p_text = colors[0]
                try:
                    fine_prompts.append(self.fine_template.format(y=cname, s=s_text, p=p_text))
                except (KeyError, ValueError):
                    fine_prompts.append(f"a photo of {cname} tissue, with {s_text} structures, and cells typically appearing {p_text}.")
            elif len(structures) > 0:
                # Use first structure if available
                s_text = structures[0]
                fine_prompts.append(f"a photo of {cname} tissue, with {s_text} structures.")
            elif len(colors) > 0:
                # Use first color if available
                p_text = colors[0]
                fine_prompts.append(f"a photo of {cname} tissue, with cells typically appearing {p_text}.")
            else:
                # Minimal fallback prompt
                fine_prompts.append(f"a photo of {cname} tissue.")
        
        return fine_prompts

    def _infer_context_length(self) -> int:
        v = getattr(self, "_context_length", None)
        if isinstance(v, int) and 0 < v < 4096:
            return int(v)
        for attr in ["context_length", "model_max_length", "max_length", "seq_length"]:
            vv = getattr(self.tokenizer, attr, None)
            if isinstance(vv, int) and 0 < vv < 4096:
                return int(vv)
        return 77

    def _pad_to_length(self, ids_list, L: int, pad_id: int = 0) -> torch.Tensor:
        out = torch.full((len(ids_list), L), int(pad_id), dtype=torch.long)
        for i, ids in enumerate(ids_list):
            ids = list(ids)[:L]
            if len(ids) > 0:
                out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        return out

    def _tokenize_texts(self, texts, device=None) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        assert isinstance(texts, (list, tuple)) and len(texts) > 0

        tok = self.tokenizer(texts)
        L = self._infer_context_length()
        pad_id = getattr(self.tokenizer, "pad_token_id", 0)

        if torch.is_tensor(tok):
            out = tok.to(torch.long)
            if out.dim() == 1:
                out = out.unsqueeze(0)
            if out.size(1) != L:
                if out.size(1) > L:
                    out = out[:, :L]
                else:
                    pad = out.new_full((out.size(0), L - out.size(1)), int(pad_id))
                    out = torch.cat([out, pad], dim=1)
            return out.to(device) if device is not None else out

        def _to_ids_list(x):
            if isinstance(x, dict) and ("input_ids" in x):
                x = x["input_ids"]
            elif hasattr(x, "input_ids"):
                x = x.input_ids

            if hasattr(x, "ids"):
                return [x.ids]

            if isinstance(x, (list, tuple)) and len(x) > 0 and hasattr(x[0], "ids"):
                return [t.ids for t in x]

            if isinstance(x, (list, tuple)) and (len(x) == 0 or isinstance(x[0], int)):
                return [list(x)]

            if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple)):
                return [list(xx) for xx in x]

            try:
                import numpy as np
                if isinstance(x, np.ndarray):
                    if x.ndim == 1:
                        return [x.tolist()]
                    if x.ndim == 2:
                        return [row.tolist() for row in x]
            except Exception:
                pass

            raise TypeError(f"Unsupported tokenizer output type: {type(x)}")

        ids_list = _to_ids_list(tok)
        out = self._pad_to_length(ids_list, L=L, pad_id=pad_id)
        if device is not None:
            out = out.to(device)
        return out

    @staticmethod
    def _dtype_str_to_torch(dtype: str):
        m = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        return m.get(str(dtype).lower(), torch.float16)

    @torch.no_grad()
    def _encode_text_nograd_chunked(
        self,
        conch_model,
        tok_ids_cpu: torch.Tensor,
        device,
        amp_dtype=torch.float16,
        chunk=64,
        desc=None,
        leave=False,
    ):
        """Encode text in chunks without gradients"""
        dev = device if isinstance(device, torch.device) else torch.device(device)
        assert tok_ids_cpu.dim() == 2, f"tok_ids must be [N,L], got {tok_ids_cpu.shape}"

        use_amp = (dev.type == "cuda") and (amp_dtype in (torch.float16, torch.bfloat16))
        N = tok_ids_cpu.size(0)
        feats = []

        it = range(0, N, int(chunk))
        if desc is not None:
            it = tqdm(it, desc=desc, leave=leave, total=(N + int(chunk) - 1) // int(chunk))

        conch_model.eval()
        for i in it:
            t = tok_ids_cpu[i:i+int(chunk)].to(dev, non_blocking=True)
            with autocast(enabled=use_amp, dtype=amp_dtype):
                e = conch_model.encode_text(t)
            e = e / (e.norm(dim=-1, keepdim=True) + 1e-12)
            feats.append(e.detach())
        return torch.cat(feats, dim=0)

    @torch.no_grad()
    def update_fine_cache(self, conch_model, device, dtype="fp16", chunk=64, show_pbar=True):
        """Update fine prompt cache"""
        assert self._ready, "PromptBank not built. Call build(...) first."
        if self._tok_fine_ids is None:
            self._tok_fine_ids = self._tokenize_texts(self._fine_texts, device=None)

        dev = device if isinstance(device, torch.device) else torch.device(device)
        amp_dtype = self._dtype_str_to_torch(dtype)

        desc = "FineCache: encode_text" if show_pbar else None
        fine_emb = self._encode_text_nograd_chunked(
            conch_model,
            tok_ids_cpu=self._tok_fine_ids,
            device=dev,
            amp_dtype=amp_dtype,
            chunk=int(chunk),
            desc=desc,
            leave=False,
        )

        self._fine_emb_cache = fine_emb.detach()
        self._fine_cache_meta = {"dtype": str(dtype).lower(), "device": str(dev), "n": int(fine_emb.size(0))}
        return self._fine_emb_cache

    def build(self, conch_model, device, enable_cache_tokens=True, purify_chunk=64, show_pbar=True):
        """Build prompt bank with purification"""
        self._context_length = int(getattr(conch_model, "context_length", 77))

        # coarse
        coarse_texts = []
        coarse_class_ids = []
        for ci, cname in enumerate(self.class_names):
            for t in self.coarse_templates:
                coarse_texts.append(t.format(cname))
                coarse_class_ids.append(ci)

        # fine
        fine_texts = []
        fine_class_ids = []
        for ci, cname in enumerate(self.class_names):
            prompts = self._generate_fine_prompts_for_class(cname)
            if len(prompts) == 0:
                # Fallback: ensure at least one prompt per class
                print(f"Warning: No fine prompts generated for class '{cname}'. Using fallback prompt.")
                print(f"  Structure bank has key: {cname in self.structure_bank}")
                print(f"  Color bank has key: {cname in self.color_bank}")
                if cname in self.structure_bank:
                    print(f"  Structure bank entries: {len(self.structure_bank[cname])}")
                if cname in self.color_bank:
                    print(f"  Color bank entries: {len(self.color_bank[cname])}")
                prompts = [f"a photo of {cname} tissue."]
            for p in prompts:
                fine_texts.append(p)
                fine_class_ids.append(ci)

        # Ensure we have at least some fine prompts
        if len(fine_texts) == 0:
            raise ValueError(
                f"No fine prompts generated for any class. "
                f"Check structure_bank and color_bank configuration. "
                f"Class names: {self.class_names}, "
                f"Structure bank keys: {list(self.structure_bank.keys())}, "
                f"Color bank keys: {list(self.color_bank.keys())}"
            )

        tok_coarse = self._tokenize_texts(coarse_texts, device=None)
        tok_fine = self._tokenize_texts(fine_texts, device=None)

        if self.purify:
            dev = device if isinstance(device, torch.device) else torch.device(device)
            conch_model.eval()

            coarse_emb_all = self._encode_text_nograd_chunked(
                conch_model,
                tok_ids_cpu=tok_coarse,
                device=dev,
                amp_dtype=torch.float16,
                chunk=int(purify_chunk),
                desc=("Purify: coarse encode" if show_pbar else None),
                leave=False,
            )

            D = coarse_emb_all.shape[-1]
            K = len(self.class_names)
            coarse_by_class = torch.zeros(K, D, device=dev)
            T = len(self.coarse_templates)
            for ci in range(K):
                s = ci * T
                e = s + T
                v = coarse_emb_all[s:e].mean(dim=0)
                v = v / (v.norm() + 1e-12)
                coarse_by_class[ci] = v

            fine_emb_all = self._encode_text_nograd_chunked(
                conch_model,
                tok_ids_cpu=tok_fine,
                device=dev,
                amp_dtype=torch.float16,
                chunk=int(purify_chunk),
                desc=("Purify: fine encode" if show_pbar else None),
                leave=False,
            )

            keep = [False] * len(fine_texts)
            fine_emb_cpu = fine_emb_all.detach().cpu()
            coarse_by_class_cpu = coarse_by_class.detach().cpu()

            class_iter = range(K)
            if show_pbar:
                class_iter = tqdm(class_iter, desc="Purify: select", leave=False)

            for ci in class_iter:
                idxs = [i for i, c in enumerate(fine_class_ids) if c == ci]
                if len(idxs) == 0:
                    continue

                E = fine_emb_cpu[idxs]
                S = E @ coarse_by_class_cpu.t()
                top2 = torch.topk(S, k=2, dim=1).values
                margin = (top2[:, 0] - top2[:, 1]).numpy()
                ok_amb = margin >= float(self.amb_margin)

                idxs2 = [idxs[j] for j in range(len(idxs)) if ok_amb[j]]
                if len(idxs2) == 0:
                    idxs2 = idxs

                selected = []
                for ii in idxs2:
                    if len(selected) == 0:
                        selected.append(ii)
                        continue
                    ei = fine_emb_cpu[ii]
                    sims = [float((ei * fine_emb_cpu[jj]).sum().item()) for jj in selected]
                    if max(sims) < float(self.red_thr):
                        selected.append(ii)
                    if len(selected) >= self.max_fine_per_class:
                        break

                if len(selected) < self.min_keep_per_class:
                    rest = [x for x in idxs2 if x not in selected]
                    for x in rest:
                        selected.append(x)
                        if len(selected) >= self.min_keep_per_class:
                            break
                    selected = selected[:max(self.min_keep_per_class, self.max_fine_per_class)]

                for ii in selected:
                    keep[ii] = True

            kept_indices = [i for i, k in enumerate(keep) if k]
            fine_texts = [fine_texts[i] for i in kept_indices]
            fine_class_ids = [fine_class_ids[i] for i in kept_indices]
            tok_fine = self._tokenize_texts(fine_texts, device=None)

            print(f"PromptBank purify: fine prompts {len(keep)} -> {len(fine_texts)} kept")

        K = len(self.class_names)
        idx_by_class = []
        for ci in range(K):
            idxs = [i for i, c in enumerate(fine_class_ids) if c == ci]
            idx_by_class.append(torch.tensor(idxs, dtype=torch.long))

        self._coarse_texts = coarse_texts
        self._fine_texts = fine_texts
        self._fine_class_ids = fine_class_ids
        self._idx_by_class_cpu = idx_by_class

        self._tok_coarse_ids = tok_coarse if enable_cache_tokens else None
        self._tok_fine_ids = tok_fine if enable_cache_tokens else None
        self._ready = True

        print(f"PromptBank built | coarse={len(coarse_texts)} | fine={len(fine_texts)}")
        for ci, cname in enumerate(self.class_names):
            cnt = sum(1 for c in fine_class_ids if c == ci)
            print(f"   - {cname:>24s}: fine={int(cnt)}")

    def _tokenize_if_needed(self):
        assert self._ready, "PromptBank not built. Call build(...)"
        if self._tok_coarse_ids is None:
            self._tok_coarse_ids = self._tokenize_texts(self._coarse_texts, device=None)
        if self._tok_fine_ids is None:
            self._tok_fine_ids = self._tokenize_texts(self._fine_texts, device=None)
        return self._tok_coarse_ids, self._tok_fine_ids

    def forward(self, conch_model, device):
        """
        Returns:
          coarse_by_class: [K,512] WITH grad
          fine_emb:        [Nf,512] NO grad (cached)
          idx_by_class:    list[tensor] (CPU)
        """
        assert self._ready, "PromptBank not built."
        tok_coarse, _ = self._tokenize_if_needed()
        tok_coarse = tok_coarse.to(device, non_blocking=True)

        # coarse WITH grad (train LoRA)
        coarse_all = conch_model.encode_text(tok_coarse)
        coarse_all = coarse_all / (coarse_all.norm(dim=-1, keepdim=True) + 1e-12)

        K = len(self.class_names)
        T = len(self.coarse_templates)
        D = coarse_all.size(-1)
        coarse_by_class = coarse_all.new_zeros((K, D))
        for ci in range(K):
            s = ci * T
            e = s + T
            v = coarse_all[s:e].mean(dim=0)
            v = v / (v.norm() + 1e-12)
            coarse_by_class[ci] = v

        # fine NO grad: use cache
        fine_emb = self._fine_emb_cache
        if fine_emb is None:
            fine_emb = self.update_fine_cache(conch_model, device=device, dtype="fp16", chunk=64, show_pbar=False)
        else:
            if fine_emb.device != torch.device(device):
                fine_emb = fine_emb.to(device, non_blocking=True)

        return coarse_by_class, fine_emb, self._idx_by_class_cpu


class KnowledgeBank(nn.Module):
    """
    Knowledge Bank using BERT for domain knowledge
    Cache ONLY BERT CLS vectors, projections trainable.
    """
    
    def __init__(self, bert_id, class_names, knowledge_texts, vision_dim=768, tokens_per_class=1,
                 freeze_bert=True, max_len=256, text_dim=512):
        super().__init__()
        self.bert_tok = AutoTokenizer.from_pretrained(bert_id)
        self.bert = AutoModel.from_pretrained(bert_id)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad_(False)

        bert_dim = self.bert.config.hidden_size
        self.proj_to_vision = nn.Linear(bert_dim, vision_dim)
        self.proj_to_text = nn.Linear(bert_dim, text_dim)

        self.class_names = list(class_names)
        self.knowledge_texts = dict(knowledge_texts)
        self.tokens_per_class = int(tokens_per_class)
        self.max_len = int(max_len)

        self.register_buffer("know_vec_bert", torch.zeros(len(self.class_names), bert_dim), persistent=True)
        self._ready = False

    @torch.no_grad()
    def build(self, device, use_web=False, timeout=5.0, show_pbar=True):
        """Build knowledge bank from texts"""
        self.eval()
        dev = device if isinstance(device, torch.device) else torch.device(device)

        it = self.class_names
        if show_pbar:
            it = tqdm(it, desc="KnowledgeBank", leave=False)

        all_vec_bert = []
        for cname in it:
            txt = self.knowledge_texts.get(cname, f"{cname} in histopathology.")
        
            tok = self.bert_tok(
                txt, padding=True, truncation=True,
                max_length=self.max_len, return_tensors="pt"
            ).to(dev)

            out = self.bert(**tok)
            cls_vec = out.last_hidden_state[:, 0, :].squeeze(0)
            all_vec_bert.append(cls_vec.detach())

        know_vec_bert = torch.stack(all_vec_bert, dim=0)
        self.know_vec_bert.copy_(know_vec_bert)
        self._ready = True
        print("KnowledgeBank built:", tuple(self.know_vec_bert.shape))

    def get_knowledge_tokens_vision(self):
        """Get knowledge tokens for vision space"""
        assert self._ready
        vis = self.proj_to_vision(self.know_vec_bert)  # [K,768]
        vis = vis / (vis.norm(dim=-1, keepdim=True) + 1e-12)
        if self.tokens_per_class > 1:
            vis = vis.repeat_interleave(self.tokens_per_class, dim=0)
        return vis

    def get_knowledge_for_labels_textspace(self):
        """Get knowledge for text space"""
        assert self._ready
        k = self.proj_to_text(self.know_vec_bert)       # [K,512]
        return k / (k.norm(dim=-1, keepdim=True) + 1e-12)

