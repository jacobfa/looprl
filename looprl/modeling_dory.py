"""Dory (normalized latent-recurrent transformer) in HF style, mirroring vLLM's reference implementation.

Parameter names match the released checkpoint so FSDP <-> vLLM weight sync needs no renaming.
The forward optionally returns per-loop quantities used by the RLTT and loop-residual (LRPG) objectives.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel

from .configuration_dory import DoryConfig

EPS = 1e-6


def justnorm(x, dim=-1):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).sum(dim, keepdim=True) + EPS)).to(x.dtype)


def slerp_alpha(a, b, alpha, alpha_scale):
    lr = (alpha.float() * alpha_scale).abs()
    ya, yb = justnorm(a).float(), justnorm(b).float()
    return justnorm(ya + lr * (yb - ya)).to(a.dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    xf = x.float()
    return (xf * cos + rotate_half(xf) * sin).to(x.dtype)


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    w = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        w = w + attention_mask[:, :, :, : key.shape[-2]]
    w = F.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(w, value).transpose(1, 2).contiguous(), w


class DoryAttention(nn.Module):
    def __init__(self, config: DoryConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        q_size, kv_size = self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.g_proj = nn.Linear(config.hidden_size, q_size, bias=False)
        self.o_proj = nn.Linear(q_size, config.hidden_size, bias=False)
        self.sqk = nn.Parameter(torch.full((q_size,), config.initializer_range))
        self.s_gate = nn.Parameter(torch.full((q_size,), config.initializer_range))
        self.sqk_factor = self.head_dim**0.5 / config.initializer_range
        g0 = config.attention_output_gate_logit_scale_init
        g0 = config.hidden_size**0.5 if g0 is None else g0
        self.s_gate_factor = g0**0.5 / config.initializer_range
        self.layer_type = config.layer_types[layer_idx]
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def forward(self, h, position_embeddings, attention_mask=None, **kwargs):
        b, t, _ = h.shape
        q = self.q_proj(h).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        gate = self.g_proj(h)
        cos, sin = position_embeddings
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        sqk = (self.sqk.float() * self.sqk_factor).view(1, self.num_heads, 1, self.head_dim)
        q = (sqk.square() * justnorm(q).float()).to(q.dtype)
        k = justnorm(k)
        impl = self.config._attn_implementation
        attn_fn = eager_attention_forward if impl in (None, "eager") else ALL_ATTENTION_FUNCTIONS[impl]
        out, _ = attn_fn(
            self, q, k, v, attention_mask, dropout=0.0, scaling=self.scaling, sliding_window=self.sliding_window, **kwargs
        )
        out = out.reshape(b, t, -1)
        sg = (self.s_gate.float() * self.s_gate_factor).square()
        out = (out.float() * torch.sigmoid(gate.float() * sg)).to(out.dtype)
        return self.o_proj(justnorm(out))


class DoryMLP(nn.Module):
    def __init__(self, config: DoryConfig):
        super().__init__()
        ffn = config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, ffn, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, config.hidden_size, bias=False)
        self.suv_gate = nn.Parameter(torch.ones(ffn))
        self.suv_up = nn.Parameter(torch.ones(ffn))
        self.suv_factor = config.hidden_size**0.5

    def forward(self, h):
        g, u = self.gate_proj(h), self.up_proj(h)
        g = (g.float() * (self.suv_gate.float() * self.suv_factor)).to(g.dtype)
        u = (u.float() * (self.suv_up.float() * self.suv_factor)).to(u.dtype)
        m = (F.silu(g.float()) * u.float()).to(h.dtype)
        return self.down_proj(justnorm(m))


class DoryDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DoryConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.is_attention = self.layer_type != "mlp"
        self.alpha_scale = config.ngpt_alpha_init / config.initializer_range
        if self.is_attention:
            self.mixer = DoryAttention(config, layer_idx)
            self.attn_alpha = nn.Parameter(torch.full((config.hidden_size,), config.initializer_range))
        else:
            self.mixer = DoryMLP(config)
            self.mlp_alpha = nn.Parameter(torch.full((config.hidden_size,), config.initializer_range))

    def forward(self, h, position_embeddings=None, attention_mask=None, **kwargs):
        if self.is_attention:
            out = self.mixer(h, position_embeddings, attention_mask=attention_mask, **kwargs)
            return slerp_alpha(h, out, self.attn_alpha, self.alpha_scale)
        return slerp_alpha(h, self.mixer(h), self.mlp_alpha, self.alpha_scale)


class DoryPreTrainedModel(PreTrainedModel):
    config: DoryConfig
    config_class = DoryConfig
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DoryDecoderLayer"]
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_attention_backend = True

    @torch.no_grad()
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear | nn.Embedding):
            module.weight.normal_(0.0, std)
        elif isinstance(module, DoryLMHead):
            module.weight.normal_(0.0, std)
            module.logit_scale.fill_(std)
        elif isinstance(module, DoryAttention):
            module.sqk.fill_(std)
            module.s_gate.fill_(std)
        elif isinstance(module, DoryMLP):
            module.suv_gate.fill_(1.0)
            module.suv_up.fill_(1.0)
        elif isinstance(module, DoryDecoderLayer):
            (module.attn_alpha if module.is_attention else module.mlp_alpha).fill_(std)


class DoryModel(DoryPreTrainedModel):
    def __init__(self, config: DoryConfig):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DoryDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.post_init()

    def _rope(self, position_ids, layer_type):
        theta = self.config.rope_parameters[layer_type]["rope_theta"]
        d = self.config.head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, device=position_ids.device, dtype=torch.float32) / d))
        freqs = position_ids[..., None].float() * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)[:, None]
        return emb.cos(), emb.sin()

    def _context(self, h, attention_mask, position_ids):
        kinds = set(self.config.layer_types) - {"mlp"}
        mask_fn = {"full_attention": create_causal_mask, "sliding_attention": create_sliding_window_causal_mask}
        ctx = {}
        for kind in kinds:
            mask = mask_fn[kind](self.config, h, attention_mask, None, position_ids)
            ctx[kind] = {"position_embeddings": self._rope(position_ids, kind), "attention_mask": mask}
        ctx["position_ids"] = position_ids
        return ctx

    def _run(self, layers, h, ctx):
        for layer in layers:
            if layer.is_attention:
                h = layer(h, position_ids=ctx["position_ids"], **ctx[layer.layer_type])
            else:
                h = layer(h)
        return h

    def run_output(self, state, ctx):
        n = self.config.n_input_layers + self.config.n_recurrent_layers
        return self._run(self.layers[n:], state, ctx)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, **kwargs):
        h = self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        if position_ids is None:
            position_ids = torch.arange(h.shape[1], device=h.device)[None].expand(h.shape[0], -1)
        ctx = self._context(h, attention_mask, position_ids)
        n_in, n_rec = self.config.n_input_layers, self.config.n_recurrent_layers
        h = self._run(self.layers[:n_in], h, ctx)
        input_h, state, states = h, h, []
        for _ in range(self.config.n_recurrent_loops):
            h = self._run(self.layers[n_in : n_in + n_rec], justnorm(state + input_h), ctx)
            state = justnorm(h)
            states.append(state)
        return self.run_output(state, ctx), states, ctx


@dataclass
class DoryCausalLMOutput(CausalLMOutputWithPast):
    loop_logp: torch.Tensor | None = None  # (B, T, L-1): log P^(l)(y_t) for the non-final loops (RLTT / deep supervision)
    div_pq: torch.Tensor | None = None  # (B, T): D(p_t || sg q_t), D = KL or JS, p = loop L, q = loop L-1 (LRPG)
    d_y: torch.Tensor | None = None  # (B, T): d_t(y_t) = Delta_t(y_t) - E_p[Delta_t], Delta = dD/dp (LRPG)
    logp_prev: torch.Tensor | None = None  # (B, T): log P^(L-1)(y_t)
    alpha_stats: torch.Tensor | None = None  # (B, T, 3): |u|^2, <u, p*d>, |p*d|^2 for fitting alpha (LRPG)


class DoryLMHead(nn.Module):
    # All head maths (incl. the chunked per-token terms) runs inside one module call, so FSDP unshards the head once per
    # call and every rank issues the same collectives no matter how many chunks its micro-batch needs.
    def __init__(self, config: DoryConfig):
        super().__init__()
        self.std, self.tau, self.chunk = config.initializer_range, config.rl_temperature, config.rl_chunk_size
        self.div = config.rl_divergence
        self.weight = nn.Parameter(torch.empty(config.vocab_size, config.hidden_size))
        self.logit_scale = nn.Parameter(torch.full((config.vocab_size,), config.initializer_range))

    def logits(self, h, temperature=1.0):
        return F.linear(h, self.weight).float() * (self.logit_scale.float() / (self.std * temperature))

    def _logp(self, h, y):
        return (torch.log_softmax(self.logits(h, self.tau), -1).gather(-1, y[:, None]).squeeze(-1),)

    def _lrpg(self, h, h_prev, y):
        logp = torch.log_softmax(self.logits(h, self.tau), -1)
        p = logp.exp()
        with torch.no_grad():
            logq = torch.log_softmax(self.logits(h_prev, self.tau), -1)
        if self.div == "js":
            logm = torch.logaddexp(logp, logq) - math.log(2)
            delta = 0.5 * (logp - logm)
            div = (p * delta).sum(-1) + 0.5 * (logq.exp() * (logq - logm)).sum(-1)
        else:
            delta = logp - logq
            div = (p * delta).sum(-1)
        with torch.no_grad():
            d = delta - (p * delta).sum(-1, keepdim=True)
            d_y = d.gather(-1, y[:, None]).squeeze(-1)
            logp_prev = logq.gather(-1, y[:, None]).squeeze(-1)
            # sufficient statistics for the residual-coefficient fit: with u = e_y - p, c = d_y u - p*d,
            # <g,c> = A (d_y |u|^2 - <u,p*d>) and |c|^2 = d_y^2 |u|^2 - 2 d_y <u,p*d> + |p*d|^2
            p_y, p2 = p.gather(-1, y[:, None]).squeeze(-1), p * p
            stats = torch.stack([1 - 2 * p_y + p2.sum(-1), p_y * d_y - (p2 * d).sum(-1), (p2 * d * d).sum(-1)], -1)
        return div, d_y, logp_prev, stats

    def forward(self, h, aux=None, y=None, h_prev=None):
        if aux is None:
            return self.logits(h)
        fn, ts = (self._logp, (h, y)) if aux == "logp" else (self._lrpg, (h, h_prev, y))
        flat = [t.reshape(-1, *t.shape[2:]) for t in ts]
        outs = [checkpoint(fn, *parts, use_reentrant=False) for parts in zip(*[t.split(self.chunk) for t in flat])]
        return [torch.cat(o).view(*h.shape[:2], *o[0].shape[1:]) for o in zip(*outs)]


class DoryForCausalLM(DoryPreTrainedModel):
    def __init__(self, config: DoryConfig):
        super().__init__(config)
        self.backbone = DoryModel(config)
        self.lm_head = DoryLMHead(config)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        rl_aux=None,
        **kwargs,
    ):
        h, states, ctx = self.backbone(input_ids, attention_mask, position_ids, inputs_embeds)
        out = DoryCausalLMOutput(logits=self.lm_head(h))
        mode, deep = self.config.rl_aux, self.config.rl_deep_supervision
        compute_aux = rl_aux if rl_aux is not None else torch.is_grad_enabled()
        if (mode is None and not deep) or not compute_aux or input_ids is None:
            return out
        assert len(states) >= 2, "per-loop objectives need n_recurrent_loops >= 2"
        y = torch.roll(input_ids, shifts=-1, dims=1)
        hs = [self.backbone.run_output(s, ctx) for s in states[:-1]] if mode == "rltt" or deep else []
        if hs:
            out.loop_logp = torch.stack([self.lm_head(hl, "logp", y)[0] for hl in hs], dim=-1)
        if mode == "lrpg":
            if hs:
                h_prev = hs[-1].detach()
            else:
                with torch.no_grad():
                    h_prev = self.backbone.run_output(states[-2].detach(), ctx)
            out.div_pq, out.d_y, out.logp_prev, out.alpha_stats = self.lm_head(h, "lrpg", y, h_prev)
        return out


__all__ = ["DoryConfig", "DoryPreTrainedModel", "DoryModel", "DoryForCausalLM", "DoryCausalLMOutput"]
