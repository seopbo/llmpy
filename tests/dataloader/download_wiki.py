import math
import os
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

from datasets import load_dataset
from tqdm.contrib.concurrent import process_map

MiB = 1024 * 1024


def calculate_proper_shard_count(dataset_size_in_bytes):
    return math.ceil(dataset_size_in_bytes / (400 * MiB))


def save_shard(idx, num_shards, output_dirpath, dset):
    shard_dataset = dset.shard(num_shards=num_shards, index=idx, contiguous=True)
    shard_filename = f"{idx:06d}.jsonl"
    shard_filepath = f"{output_dirpath}/{shard_filename}"
    shard_dataset.to_json(shard_filepath, force_ascii=False)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--lang", type=str, choices=["ko", "en"])
    parser.add_argument("--num_proc", type=int, default=os.cpu_count() // 2)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    dset = load_dataset("wikimedia/wikipedia", f"20231101.{args.lang}", num_proc=args.num_proc)
    dset = dset["train"]
    dset = dset.map(
        lambda record: {"text": record["title"] + "\n\n" + record["text"]},
        remove_columns=dset.column_names,
        num_proc=args.num_proc,
    )
    num_shards = calculate_proper_shard_count(dset._estimate_nbytes())
    output_dirpath = Path(__file__).parent / "datasets" / "wiki" / args.lang

    save_shard_ = partial(
        save_shard,
        num_shards=num_shards,
        output_dirpath=output_dirpath,
        dset=dset,
    )

    _ = process_map(
        save_shard_,
        range(num_shards),
        max_workers=args.num_proc,
        chunksize=max(math.ceil(num_shards // args.num_proc), 1),
    )


if __name__ == "__main__":
    main()
