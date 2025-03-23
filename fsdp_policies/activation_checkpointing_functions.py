# ref: https://github.com/pytorch/examples/blob/main/distributed/FSDP/policies/activation_checkpointing_functions.py
from functools import partial

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from transformers.models.llama.modeling_llama import LlamaAttention

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    offload_to_cpu=False,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

check_fn = lambda submodule: isinstance(submodule, LlamaAttention)


def apply_fsdp_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying fdsp activation checkpointing...")

    apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
