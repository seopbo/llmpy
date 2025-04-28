from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

from webdataset import WebDataset


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--lang", type=str, choices=["ko", "en"])
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    base_dirpath = Path(__file__).parent
    lang_dirpath = base_dirpath / args.lang

    dataset = (
        WebDataset(urls=f"{str(lang_dirpath)}/0000{{0..5}}.tar")  # 25 shards
        .decode()  # Automagically decode files
        .shuffle(size=1000)  # Shuffle on-the-fly in a buffer
        .batched(batchsize=1)  # Create batches
    )

    batch: Dict = next(iter(dataset))
    print(batch)


if __name__ == "__main__":
    main()
