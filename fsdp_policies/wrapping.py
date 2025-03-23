# ref: https://github.com/pytorch/examples/blob/main/distributed/FSDP/policies/wrapping.py
# holds various wrapping policies for fsdp


import functools
from typing import Type

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
    wrap,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


def get_size_policy(min_params=1e8):
    num_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_params)
    return num_wrap_policy


def get_llama_wrapper():
    """we register our main layer class and use the fsdp transformer wrapping policy
    ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers
    """
    # ====   use new transformer wrapper

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer,
        },
    )

    return llama_auto_wrap_policy
