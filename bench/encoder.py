"""MiniLM, reimplemented twice, so the ANE layout question can be measured.

Why not just convert the HuggingFace model? Two reasons, one incidental and one
fundamental.

The incidental one: transformers 5.x emits `aten::Int` operations from its
dynamic attention-mask machinery (`q_length`, `kv_length`, `batch_size`), and
coremltools cannot convert them. Pinning old library versions would paper over
this, but a benchmark nobody can reproduce in two years is not worth much.

The fundamental one: a stock BERT graph does not run well on the Neural Engine
regardless of conversion success. Apple's `ml-ane-transformers` work documents
why, and the fixes are architectural rather than cosmetic:

  * **(B, C, 1, S) layout instead of (B, S, C).** The ANE is built around a
    channels-first 4D tensor. Feeding it the transformer-conventional layout
    forces transposes on every block.
  * **1x1 Conv2d instead of Linear.** Same arithmetic, but convolution is the
    ANE's native primitive; Linear tends to fall back.
  * **Per-head split attention.** One large batched matmul over all heads
    creates tensors that exceed ANE working-set limits. Apple's fix is to run
    heads as separate small matmuls -- more operations, but each one stays
    resident.
  * **LayerNorm over the channel axis**, written out explicitly rather than via
    `nn.LayerNorm`, which normalises the last axis and would be wrong here.

`ReferenceEncoder` is the honest control: correct, conventional, conversion-safe.
`ANEEncoder` is the same weights in Apple's layout. Both load from the same
checkpoint and are verified against HuggingFace output, so any latency gap is
attributable to layout alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class BertConfig:
    vocab_size: int = 30522
    hidden_size: int = 384
    num_layers: int = 6
    num_heads: int = 12
    intermediate_size: int = 1536
    max_position: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @classmethod
    def from_hf(cls, cfg) -> "BertConfig":
        return cls(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_hidden_layers,
            num_heads=cfg.num_attention_heads,
            intermediate_size=cfg.intermediate_size,
            max_position=cfg.max_position_embeddings,
            type_vocab_size=cfg.type_vocab_size,
            layer_norm_eps=getattr(cfg, "layer_norm_eps", 1e-12),
        )


# ==========================================================================
# Reference implementation -- conventional (B, S, C)
# ==========================================================================


class ReferenceLayer(nn.Module):
    def __init__(self, c: BertConfig):
        super().__init__()
        h = c.hidden_size
        self.num_heads, self.head_dim = c.num_heads, c.head_dim
        self.query, self.key, self.value = nn.Linear(h, h), nn.Linear(h, h), nn.Linear(h, h)
        self.attn_out = nn.Linear(h, h)
        self.attn_ln = nn.LayerNorm(h, eps=c.layer_norm_eps)
        self.intermediate = nn.Linear(h, c.intermediate_size)
        self.output = nn.Linear(c.intermediate_size, h)
        self.out_ln = nn.LayerNorm(h, eps=c.layer_norm_eps)
        self.act = nn.GELU()

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        # unflatten/permute rather than view(b, s, ...): reading x.shape under
        # torch.jit.trace emits aten::Int, which coremltools cannot convert.
        # Every shape query in this file is avoided for that reason.
        return x.unflatten(-1, (self.num_heads, self.head_dim)).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q, k, v = self._heads(self.query(x)), self._heads(self.key(x)), self._heads(self.value(x))

        scores = (q @ k.transpose(-1, -2)) * (self.head_dim ** -0.5) + mask
        ctx = (scores.softmax(dim=-1) @ v).permute(0, 2, 1, 3).flatten(-2)

        x = self.attn_ln(x + self.attn_out(ctx))
        return self.out_ln(x + self.output(self.act(self.intermediate(x))))


class ReferenceEncoder(nn.Module):
    """Conventional layout. Correct, convertible, and not ANE-friendly."""

    layout = "b_s_c"

    def __init__(self, c: BertConfig, seq_len: int | None = None):
        super().__init__()
        self.cfg = c
        self.seq_len = seq_len
        self.word_emb = nn.Embedding(c.vocab_size, c.hidden_size)
        self.pos_emb = nn.Embedding(c.max_position, c.hidden_size)
        self.type_emb = nn.Embedding(c.type_vocab_size, c.hidden_size)
        self.emb_ln = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.layers = nn.ModuleList(ReferenceLayer(c) for _ in range(c.num_layers))
        # Sized to seq_len when fixed, so forward() never has to slice by a
        # traced shape. Dynamic slicing is what emits aten::Int.
        n_pos = seq_len or c.max_position
        self.register_buffer(
            "position_ids", torch.arange(n_pos).unsqueeze(0), persistent=False
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pos = self.position_ids if self.seq_len else self.position_ids[:, : input_ids.shape[1]]
        x = self.word_emb(input_ids) + self.pos_emb(pos) + self.type_emb(torch.zeros_like(input_ids))
        x = self.emb_ln(x)

        m = attention_mask.to(x.dtype)
        additive = (1.0 - m)[:, None, None, :] * -1e4

        for layer in self.layers:
            x = layer(x, additive)

        mask_e = m.unsqueeze(-1)
        pooled = (x * mask_e).sum(dim=1) / mask_e.sum(dim=1).clamp(min=1e-9)
        return pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-12)


# ==========================================================================
# ANE-optimised implementation -- (B, C, 1, S)
# ==========================================================================


class LayerNormANE(nn.Module):
    """LayerNorm over the channel axis of a (B, C, 1, S) tensor.

    `nn.LayerNorm` normalises the trailing axis, which in this layout is the
    sequence -- silently wrong rather than loudly broken, so it is written out.
    """

    def __init__(self, channels: int, eps: float = 1e-12):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=1, keepdim=True)
        denom = (centered * centered).mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        return centered * denom * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class ANELayer(nn.Module):
    def __init__(self, c: BertConfig):
        super().__init__()
        h = c.hidden_size
        self.num_heads, self.head_dim = c.num_heads, c.head_dim
        self.scale = c.head_dim ** -0.5
        self.query, self.key, self.value = nn.Conv2d(h, h, 1), nn.Conv2d(h, h, 1), nn.Conv2d(h, h, 1)
        self.attn_out = nn.Conv2d(h, h, 1)
        self.attn_ln = LayerNormANE(h, c.layer_norm_eps)
        self.intermediate = nn.Conv2d(h, c.intermediate_size, 1)
        self.output = nn.Conv2d(c.intermediate_size, h, 1)
        self.out_ln = LayerNormANE(h, c.layer_norm_eps)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q, k, v = self.query(x), self.key(x), self.value(x)

        # Heads are split into separate small matmuls rather than one batched
        # matmul. This is the ANE working-set constraint, not a stylistic choice.
        mh_q = q.split(self.head_dim, dim=1)                    # (B, hd, 1, S)
        mh_k = k.transpose(1, 3).split(self.head_dim, dim=3)    # (B, S, 1, hd)
        mh_v = v.split(self.head_dim, dim=1)                    # (B, hd, 1, S)

        weights = [
            torch.einsum("bchq,bkhc->bkhq", qi, ki) * self.scale
            for qi, ki in zip(mh_q, mh_k)
        ]
        weights = [(w + mask).softmax(dim=1) for w in weights]
        ctx = torch.cat(
            [torch.einsum("bkhq,bchk->bchq", wi, vi) for wi, vi in zip(weights, mh_v)],
            dim=1,
        )

        x = self.attn_ln(x + self.attn_out(ctx))
        return self.out_ln(x + self.output(self.act(self.intermediate(x))))


class ANEEncoder(nn.Module):
    """Apple's ml-ane-transformers layout. Identical weights, different shape."""

    layout = "b_c_1_s"

    def __init__(self, c: BertConfig, seq_len: int | None = None):
        super().__init__()
        self.cfg = c
        self.seq_len = seq_len
        self.word_emb = nn.Embedding(c.vocab_size, c.hidden_size)
        self.pos_emb = nn.Embedding(c.max_position, c.hidden_size)
        self.type_emb = nn.Embedding(c.type_vocab_size, c.hidden_size)
        self.emb_ln = LayerNormANE(c.hidden_size, c.layer_norm_eps)
        self.layers = nn.ModuleList(ANELayer(c) for _ in range(c.num_layers))
        n_pos = seq_len or c.max_position
        self.register_buffer(
            "position_ids", torch.arange(n_pos).unsqueeze(0), persistent=False
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pos = self.position_ids if self.seq_len else self.position_ids[:, : input_ids.shape[1]]
        x = self.word_emb(input_ids) + self.pos_emb(pos) + self.type_emb(torch.zeros_like(input_ids))
        x = x.permute(0, 2, 1).unsqueeze(2)  # (B, S, C) -> (B, C, 1, S)
        x = self.emb_ln(x)

        # unsqueeze rather than view(m.shape[0], ...) -- same aten::Int problem.
        m = attention_mask.to(x.dtype)
        additive = ((1.0 - m) * -1e4).unsqueeze(-1).unsqueeze(-1)  # (B, S_k, 1, 1)

        for layer in self.layers:
            x = layer(x, additive)

        mask_c = m.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, S)
        pooled = (x * mask_c).sum(dim=3, keepdim=True) / mask_c.sum(dim=3, keepdim=True).clamp(min=1e-9)
        pooled = pooled.flatten(1)
        return pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-12)


# ==========================================================================
# Cross-encoders -- same bodies, a CLS head instead of mean pooling
# ==========================================================================
#
# The reranker (ms-marco-MiniLM-L6) is a BertForSequenceClassification: the
# exact 6-layer/384-dim body above plus a pooler (CLS token -> dense -> tanh)
# and a 1-logit classifier. Two differences matter for conversion:
#
#   * token_type_ids are a real input. A cross-encoder reads "query [SEP]
#     passage" and the segment embedding is how it tells the two apart --
#     zeroing it (as the embedders above do) would lobotomise the model.
#   * The workload is a *batch*: reranking scores 50 pairs per query, so the
#     interesting fixed shape is (50, seq), not (1, seq). Whether the ANE
#     keeps a batch-50 graph resident is precisely the open question the
#     conversion answers.


class ReferenceCrossEncoder(nn.Module):
    """Conventional layout; logits out. One row per (query, passage) pair."""

    layout = "b_s_c"

    def __init__(self, c: BertConfig, seq_len: int | None = None):
        super().__init__()
        self.cfg = c
        self.seq_len = seq_len
        self.word_emb = nn.Embedding(c.vocab_size, c.hidden_size)
        self.pos_emb = nn.Embedding(c.max_position, c.hidden_size)
        self.type_emb = nn.Embedding(c.type_vocab_size, c.hidden_size)
        self.emb_ln = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.layers = nn.ModuleList(ReferenceLayer(c) for _ in range(c.num_layers))
        self.pooler = nn.Linear(c.hidden_size, c.hidden_size)
        self.classifier = nn.Linear(c.hidden_size, 1)
        n_pos = seq_len or c.max_position
        self.register_buffer(
            "position_ids", torch.arange(n_pos).unsqueeze(0), persistent=False
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        pos = self.position_ids if self.seq_len else self.position_ids[:, : input_ids.shape[1]]
        x = self.word_emb(input_ids) + self.pos_emb(pos) + self.type_emb(token_type_ids)
        x = self.emb_ln(x)

        m = attention_mask.to(x.dtype)
        additive = (1.0 - m)[:, None, None, :] * -1e4
        for layer in self.layers:
            x = layer(x, additive)

        cls = x[:, 0]  # constant index -- trace-safe
        pooled = torch.tanh(self.pooler(cls))
        return self.classifier(pooled).squeeze(-1)


class ANECrossEncoder(nn.Module):
    """Apple layout; the head stays in (B, C, 1, S) until the final flatten."""

    layout = "b_c_1_s"

    def __init__(self, c: BertConfig, seq_len: int | None = None):
        super().__init__()
        self.cfg = c
        self.seq_len = seq_len
        self.word_emb = nn.Embedding(c.vocab_size, c.hidden_size)
        self.pos_emb = nn.Embedding(c.max_position, c.hidden_size)
        self.type_emb = nn.Embedding(c.type_vocab_size, c.hidden_size)
        self.emb_ln = LayerNormANE(c.hidden_size, c.layer_norm_eps)
        self.layers = nn.ModuleList(ANELayer(c) for _ in range(c.num_layers))
        self.pooler = nn.Conv2d(c.hidden_size, c.hidden_size, 1)
        self.classifier = nn.Conv2d(c.hidden_size, 1, 1)
        n_pos = seq_len or c.max_position
        self.register_buffer(
            "position_ids", torch.arange(n_pos).unsqueeze(0), persistent=False
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        pos = self.position_ids if self.seq_len else self.position_ids[:, : input_ids.shape[1]]
        x = self.word_emb(input_ids) + self.pos_emb(pos) + self.type_emb(token_type_ids)
        x = x.permute(0, 2, 1).unsqueeze(2)  # (B, C, 1, S)
        x = self.emb_ln(x)

        m = attention_mask.to(x.dtype)
        additive = ((1.0 - m) * -1e4).unsqueeze(-1).unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, additive)

        cls = x[:, :, :, 0:1]  # (B, C, 1, 1); constant slice, trace-safe
        pooled = torch.tanh(self.pooler(cls))
        return self.classifier(pooled).flatten(1).squeeze(-1)


def load_ce_weights(module: nn.Module, state: dict[str, torch.Tensor]) -> nn.Module:
    """Populate a cross-encoder from a BertForSequenceClassification checkpoint.

    The body reuses `load_weights` after stripping the `bert.` prefix; only the
    pooler and classifier are new.
    """
    body = {
        k.removeprefix("bert."): v for k, v in state.items() if k.startswith("bert.")
    }
    as_conv = module.layout == "b_c_1_s"
    load_weights(module, body)
    _copy_linear(
        module.pooler, state["bert.pooler.dense.weight"], state["bert.pooler.dense.bias"], as_conv=as_conv
    )
    _copy_linear(
        module.classifier, state["classifier.weight"], state["classifier.bias"], as_conv=as_conv
    )
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


DEFAULT_CE_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def build_ce(
    kind: str = "reference",
    model_name: str = DEFAULT_CE_MODEL,
    seq_len: int | None = None,
) -> nn.Module:
    import warnings

    warnings.filterwarnings("ignore")
    from transformers import AutoModelForSequenceClassification

    hf = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    cfg = BertConfig.from_hf(hf.config)
    cls = {"reference": ReferenceCrossEncoder, "ane": ANECrossEncoder}[kind]
    return load_ce_weights(cls(cfg, seq_len=seq_len), hf.state_dict())


# ==========================================================================
# Weight loading
# ==========================================================================


def _copy_linear(dst: nn.Module, w: torch.Tensor, b: torch.Tensor, *, as_conv: bool) -> None:
    """Copy Linear weights into either a Linear or a 1x1 Conv2d.

    A 1x1 convolution is a linear map with the kernel axes appended, so the
    weight only needs reshaping from (out, in) to (out, in, 1, 1).
    """
    with torch.no_grad():
        dst.weight.copy_(w.view(*w.shape, 1, 1) if as_conv else w)
        dst.bias.copy_(b)


def load_weights(module: nn.Module, state: dict[str, torch.Tensor]) -> nn.Module:
    as_conv = getattr(module, "layout", "b_s_c") == "b_c_1_s"
    with torch.no_grad():
        module.word_emb.weight.copy_(state["embeddings.word_embeddings.weight"])
        module.pos_emb.weight.copy_(state["embeddings.position_embeddings.weight"])
        module.type_emb.weight.copy_(state["embeddings.token_type_embeddings.weight"])
        module.emb_ln.weight.copy_(state["embeddings.LayerNorm.weight"])
        module.emb_ln.bias.copy_(state["embeddings.LayerNorm.bias"])

        for i, layer in enumerate(module.layers):
            p = f"encoder.layer.{i}."
            for name, key in (
                ("query", "attention.self.query"),
                ("key", "attention.self.key"),
                ("value", "attention.self.value"),
                ("attn_out", "attention.output.dense"),
                ("intermediate", "intermediate.dense"),
                ("output", "output.dense"),
            ):
                _copy_linear(
                    getattr(layer, name),
                    state[p + key + ".weight"],
                    state[p + key + ".bias"],
                    as_conv=as_conv,
                )
            layer.attn_ln.weight.copy_(state[p + "attention.output.LayerNorm.weight"])
            layer.attn_ln.bias.copy_(state[p + "attention.output.LayerNorm.bias"])
            layer.out_ln.weight.copy_(state[p + "output.LayerNorm.weight"])
            layer.out_ln.bias.copy_(state[p + "output.LayerNorm.bias"])

    module.eval()
    for prm in module.parameters():
        prm.requires_grad_(False)
    return module


def build(
    kind: str = "reference",
    model_name: str = DEFAULT_MODEL,
    seq_len: int | None = None,
) -> nn.Module:
    """Construct `reference` or `ane` and populate it from the HF checkpoint.

    `seq_len` fixes the sequence length so the traced graph contains no shape
    queries. Required for CoreML conversion; leave None for flexible eager use.
    """
    import warnings

    warnings.filterwarnings("ignore")
    from transformers import AutoModel

    hf = AutoModel.from_pretrained(model_name)
    hf.eval()
    cfg = BertConfig.from_hf(hf.config)
    cls = {"reference": ReferenceEncoder, "ane": ANEEncoder}[kind]
    return load_weights(cls(cfg, seq_len=seq_len), hf.state_dict())


def load_tokenizer(model_name: str = DEFAULT_MODEL):
    import warnings

    warnings.filterwarnings("ignore")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def reference_hf(model_name: str = DEFAULT_MODEL):
    """The HuggingFace model plus mean pooling, used only as a correctness oracle."""
    import warnings

    warnings.filterwarnings("ignore")
    from transformers import AutoModel

    hf = AutoModel.from_pretrained(model_name).eval()

    def run(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = hf(input_ids=input_ids.long(), attention_mask=attention_mask.long())
        h = out.last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-12)

    return run
