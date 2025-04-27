from pathlib import Path

from datasets import load_dataset
from webdataset import ShardWriter

ds = load_dataset("wikimedia/wikipedia", "20231101.ko")
ds = ds["train"]
ds[0]["title"] + "\n\n" + ds[0]["text"]
pattern = "/data/nick_722/workspace/llmpy/tests/energon/kowiki/%05d.tar"
parent_dirpath = Path(pattern).parent
parent_dirpath.mkdir(exist_ok=True)

with ShardWriter(pattern) as shard_writer:
    for idx, record in enumerate(ds):
        key = str(idx)
        text = record["title"] + "\n\n" + record["text"]
        sample = {
            "__key__": key,
            "text": text,
        }
        shard_writer.write(sample)
