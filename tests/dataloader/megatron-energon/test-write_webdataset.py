import os
from argparse import ArgumentParser
from pathlib import Path

from datasets import load_dataset
from webdataset import ShardWriter


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--lang", type=str, choices=["ko", "en"])
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    base_dirpath = Path(__file__).parent
    lang_dirpath = base_dirpath / args.lang

    if not lang_dirpath.exists():
        lang_dirpath.mkdir()
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{args.lang}", num_proc=os.cpu_count() // 2)
    ds = ds["train"]

    pattern = f"{str(lang_dirpath)}/%05d.tar"

    with ShardWriter(pattern) as shard_writer:
        for idx, record in enumerate(ds):
            key = str(idx)
            text = record["title"] + "\n\n" + record["text"]
            sample = {
                "__key__": key,
                "text": text,
            }
            shard_writer.write(sample)


if __name__ == "__main__":
    main()
