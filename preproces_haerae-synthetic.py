from datasets import load_dataset
from utils import DecoderProcessor, calculate_proper_chunk_count


def main():
    model_name_or_path = "/data/lm-old_project_language-model_732/rw/lmt/checkpoints/hf/meta-llama-3.2-1b"
    raw_ds = load_dataset("/data/nick_722/workspace/llmpy/raw_datasets/haerae-synthetic/data", num_proc=16)["train"]
    raw_ds = raw_ds.select_columns("text")

    processor = DecoderProcessor(model_name_or_path, max_length=8192)
    ds = raw_ds.map(
        lambda examples: processor(examples["text"]), num_proc=16, batched=True, remove_columns=raw_ds.column_names
    )
    num_shards = calculate_proper_chunk_count(ds._estimate_nbytes())
    output_path_template = "/data/nick_722/workspace/llmpy/datasets/haerae-synthetic-{index:05d}.parquet.zstd"

    for index in range(num_shards):
        shard = ds.shard(index=index, num_shards=num_shards, contiguous=True)
        shard.to_parquet(output_path_template.format(index=index), compression="zstd")


if __name__ == "__main__":
    main()
