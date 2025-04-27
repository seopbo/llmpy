from typing import Dict

from webdataset import WebDataset

dataset = (
    WebDataset(
        urls="/data/nick_722/workspace/llmpy/tests/energon/kowiki/0000{0..6}.tar"
    )  # 25 shards
    .decode()  # Automagically decode files
    .shuffle(size=1000)  # Shuffle on-the-fly in a buffer
    .batched(batchsize=1)  # Create batches
)


batch: Dict = next(iter(dataset))
