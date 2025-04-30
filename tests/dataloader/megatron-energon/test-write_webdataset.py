import os
import json
from argparse import ArgumentParser
from pathlib import Path
from tqdm.contrib.concurrent import process_map
from functools import partial
from webdataset import TarWriter
from typing import Dict, Generator


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--lang", type=str, choices=["ko", "en"])
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() // 2)
    args = parser.parse_args()
    return args


def get_jsonl_generator(input_filepath: Path) -> Generator[Dict[str, str], None, None]:
    with open(input_filepath, "r", encoding="utf-8") as io:
        for line_idx, jsonl_line in enumerate(io):
            jsonl_dict = json.loads(jsonl_line)
            jsonl_dict["__key__"] = f"{str(input_filepath)}/{line_idx}"
            yield jsonl_dict


def write_tar(input_filepath: Path, output_dirpath: Path) -> None:
    filename = input_filepath.stem
    with TarWriter(f"{output_dirpath}/{filename}.tar") as writer:
        for sample in get_jsonl_generator(input_filepath):
            writer.write(sample)


def main():
    args = get_args()
    input_dirpath = Path(__file__).parent.parent / "datasets" / "wiki" / args.lang
    output_dirpath = Path(__file__).parent / "datasets" / "wiki" / args.lang

    if not output_dirpath.exists():
        output_dirpath.mkdir(parents=True, exist_ok=True)

    list_of_input_filepaths = [input_filepath for input_filepath in input_dirpath.iterdir()]

    write_tar_ = partial(write_tar, output_dirpath=output_dirpath)
    _ = process_map(write_tar_, list_of_input_filepaths, max_workers=args.num_workers)


if __name__ == "__main__":
    main()
