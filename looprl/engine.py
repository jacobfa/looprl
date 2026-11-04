"""Surfaces Dory's per-loop RL auxiliaries (loop_logp, kl_pq, d_y, logp_prev, alpha_stats) through verl's FSDP engine."""

import torch

from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

AUX_KEYS = ("loop_logp", "kl_pq", "d_y", "logp_prev", "alpha_stats")
_orig_prepare_model_outputs = FSDPEngineWithLMHead.prepare_model_outputs


def prepare_model_outputs(self, output, output_args, micro_batch, logits_processor_func):
    model_output = _orig_prepare_model_outputs(self, output, output_args, micro_batch, logits_processor_func)
    aux = {k: getattr(output, k, None) for k in AUX_KEYS}
    aux = {k: v for k, v in aux.items() if v is not None}
    if not aux:
        return model_output
    cu_seqlens = micro_batch["input_ids"].offsets()
    if tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True):
        for k, v in aux.items():
            v = self._gather_and_unpad_packed(v.squeeze(0), output_args["pad_size"])
            model_output[k] = torch.nested.nested_tensor_from_jagged(v, cu_seqlens)
    else:
        lens = cu_seqlens.diff().tolist()
        for k, v in aux.items():
            v = torch.cat([row[:n] for row, n in zip(v.unbind(0), lens)])
            model_output[k] = torch.nested.nested_tensor_from_jagged(v, cu_seqlens)
    return model_output


def patch():
    FSDPEngineWithLMHead.prepare_model_outputs = prepare_model_outputs
