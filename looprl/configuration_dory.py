from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig


@strict(accept_kwargs=True)
class DoryConfig(PreTrainedConfig):
    model_type = "dory"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 131072
    hidden_size: int = 2560
    num_hidden_layers: int = 72
    intermediate_size: int = 7168
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    initializer_range: float = 0.0177
    sliding_window: int | None = 1024
    rope_parameters: dict | None = None
    block_types: list[str] | None = None  # per physical layer: sliding_attention | full_attention | mlp
    ngpt_alpha_init: float = 0.05
    attention_output_gate_logit_scale_init: float | None = None
    mlp_bias: bool = False
    attention_bias: bool = False
    n_input_layers: int = 16
    n_recurrent_layers: int | None = None
    n_output_layers: int = 16
    n_recurrent_loops: int = 2
    recurrent_kv_cache_mode: str = "per_loop"
    use_cache: bool = False
    pad_token_id: int | None = 11
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 11
    # RL auxiliaries computed by the training forward: None | "rltt" | "lrpg"; deep supervision adds per-loop log-probs
    rl_aux: str | None = None
    rl_deep_supervision: bool = False
    rl_temperature: float = 1.0
    rl_chunk_size: int = 2048

    def __post_init__(self, **kwargs):
        if self.block_types is None:
            self.block_types = [
                "mlp" if i % 2 else "full_attention" if i % 8 == 6 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.n_recurrent_layers is None:
            self.n_recurrent_layers = self.num_hidden_layers - self.n_input_layers - self.n_output_layers
        super().__post_init__(**kwargs)

    @property
    def layer_types(self):
        return self.block_types

    @layer_types.setter
    def layer_types(self, value):
        self.block_types = value

    def convert_rope_params_to_dict(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
                "full_attention": {"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 1.0},
            }
        kwargs.pop("rope_theta", None)
        kwargs.pop("rope_scaling", None)
        return kwargs

    def standardize_rope_params(self):
        pass

    def validate_layer_type(self):
        valid = {"sliding_attention", "full_attention", "mlp"}
        if any(t not in valid for t in self.layer_types) or len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must have num_hidden_layers entries from {sliding_attention, full_attention, mlp}")

    def validate_architecture(self):
        if self.n_input_layers + self.n_recurrent_layers + self.n_output_layers != self.num_hidden_layers:
            raise ValueError("input + recurrent + output layers must sum to num_hidden_layers")
        if self.n_recurrent_loops < 1 or self.recurrent_kv_cache_mode not in ("per_loop", "last_loop"):
            raise ValueError("invalid n_recurrent_loops / recurrent_kv_cache_mode")
        if self.rl_aux not in (None, "rltt", "lrpg"):
            raise ValueError(f"unknown rl_aux {self.rl_aux}")

    def validate_rope(self):
        for kind in set(self.layer_types) - {"mlp"}:
            if kind not in self.rope_parameters or self.rope_parameters[kind].get("rope_theta", 0) <= 0:
                raise ValueError(f"rope_parameters[{kind}] must define a positive rope_theta")


__all__ = ["DoryConfig"]
