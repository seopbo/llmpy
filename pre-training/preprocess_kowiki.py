# https://github.com/webdataset/webdataset/blob/main/examples/generate-text-dataset.ipynb
import webdataset as wds
from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.ko")

ds["train"][0]
output_filepath = "/data/nick_722/workspace/llmpy/pre-training/kowiki.tar"
example_text = ds["train"][0]["text"]
example_key = ds["train"][0]["url"]
list_of_texts = ["안녕", "내 이름은", "김보섭이야"]
with wds.TarWriter(output_filepath) as output:
    example = {"__key__": example_key, "text": example_text}
    output.write(example)
