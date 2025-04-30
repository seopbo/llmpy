import time
import torch
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List
from pathlib import Path

from megatron.energon import (
    DefaultTaskEncoder,
    TextSample,
    WorkerConfig,
    batch_list,
    get_loader,
    get_train_dataset,
)
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class LanguageModelingSample:
    # (l,)
    input_ids: List[int]


@dataclass
class LanguageModelingRawBatch:
    # (n, l)
    input_ids: List[List[int]]


@dataclass
class LanguageModelingBatch:
    # (n, l)
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


# All the typing is optional
class LanguageModelingTaskEncoderV2(
    DefaultTaskEncoder[
        TextSample,
        LanguageModelingSample,
        LanguageModelingRawBatch,
        LanguageModelingBatch,
    ]
):
    """A simple task encoder for language modeling."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        reset_attention_mask: bool = False,
    ):
        # Specify the batch_type for default batching (batching is performed here "manually" by overwriting the `batch`
        # method)
        super().__init__(batch_type=LanguageModelingRawBatch)
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._reset_attention_mask = reset_attention_mask
        self._buffer = []

    def encode_sample(self, sample: TextSample) -> TextSample:
        return sample

    def select_samples_to_pack(self, samples: List[TextSample]) -> List[List[LanguageModelingSample]]:
        list_of_texts = [sample.text for sample in samples]
        outputs = self._tokenizer(list_of_texts, add_special_tokens=False, return_attention_mask=False)
        list_of_input_ids = outputs["input_ids"]

        groups = []

        for input_ids in list_of_input_ids:
            input_ids += [self._tokenizer.eos_token_id]
            self._buffer.extend(input_ids)

            while len(self._buffer) >= self._max_length:
                chunk_ids = self._buffer[: self._max_length]
                groups.append([LanguageModelingSample(input_ids=chunk_ids)])
                self._buffer = self._buffer[self._max_length :]
        return groups

    def pack_selected_samples(self, samples: List[LanguageModelingSample]) -> LanguageModelingSample:
        return samples[0]

    def batch(self, samples: List[LanguageModelingSample]) -> LanguageModelingRawBatch:
        return self._batch(
            samples,
            result_type=LanguageModelingRawBatch,
            actions={"input_ids": batch_list},
        )

    def encode_batch(self, batch_data: LanguageModelingRawBatch) -> LanguageModelingBatch:
        input_ids = torch.tensor(batch_data.input_ids)
        attention_mask = torch.zeros_like(input_ids)

        if self._reset_attention_mask:
            attention_mask[input_ids != self._tokenizer.eos_token_id] = 1
        else:
            attention_mask[...] = 1
        return LanguageModelingBatch(input_ids=input_ids, attention_mask=attention_mask)


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--eds_dirpath",
        type=str,
    )
    parser.add_argument("--max_sequence_length", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--reset_attention_mask", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    simple_worker_config = WorkerConfig(rank=0, world_size=1, num_workers=4)
    pretrained_tokenizer_model_name_or_path = str(Path(__file__).parent.parent / "tokenizers" / "llama3")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_model_name_or_path)
    task_encoder = LanguageModelingTaskEncoderV2(
        tokenizer,
        max_length=args.max_sequence_length,
        reset_attention_mask=args.reset_attention_mask,
    )

    train_ds = get_train_dataset(
        args.eds_dirpath,
        batch_size=args.batch_size,
        shuffle_buffer_size=None,
        max_samples_per_sequence=None,
        packing_buffer_size=1000,
        worker_config=simple_worker_config,
        task_encoder=task_encoder,
    )
    train_dl = get_loader(train_ds)

    start_time = time.time()
    for idx, batch in enumerate(train_dl):
        print(idx)
        print(batch.input_ids, batch.input_ids.size())
        print(batch.attention_mask, batch.attention_mask.size())

        if idx == 100:
            break
    end_time = time.time()
    print(f"elapsed time: {end_time - start_time}")


if __name__ == "__main__":
    main()
