from argparse import ArgumentParser

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from transformers.optimization import AdamW, get_cosine_with_min_lr_schedule_with_warmup

from datasets import load_dataset


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dirpath", type=str, default="/data/nick_722/workspace/llmpy/datasets")
    parser.add_argument("--save_dirpath", type=str, default="/data/nick_722/workspace/llmpy/checkpoints/base")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/data/nick_722/hf_assets/llama-3.2-1b-instruct",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--training_steps", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--wd", type=float, default=1e-1)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--torch_empty_cache_every_steps", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # prepare datasets
    ds = load_dataset("parquet", data_dir=args.data_dirpath, num_proc=8)["train"]
    ds = ds.train_test_split(test_size=0.05, seed=42)
    train_ds = ds["train"].with_format("torch")
    valid_ds = ds["test"].with_format("torch")

    # prepare assets
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")

    config = AutoConfig.from_pretrained(args.pretrained_model_name_or_path)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    # model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
    model.cuda()
    torch.cuda.empty_cache()

    if args.gradient_checkpointing:

        def apply_checkpoint(module):
            orig_forward = module.forward

            def wrapped_forward(*inputs, **kwargs):
                def custom_forward(*inputs):
                    return orig_forward(*inputs, **kwargs)

                return checkpoint(custom_forward, *inputs, use_reentrant=False)

            module.forward = wrapped_forward

        for layer in model.model.layers:
            apply_checkpoint(layer.self_attn)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.per_device_train_batch_size,
        drop_last=True,
        shuffle=True,
        collate_fn=collate_fn,
    )
    optimizer = AdamW(
        params=model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.wd,
    )

    estimated_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    _training_steps = len(train_dl.dataset) // estimated_batch_size
    _num_epochs = (args.training_steps // _training_steps) + 1

    scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.training_steps, min_lr=args.min_lr
    )

    # training
    train_steps = 0
    model.train()
    optimizer.zero_grad()

    pbar = tqdm(range(args.training_steps))
    for _epoch in range(_num_epochs):
        for mb_train_index, train_mb in enumerate(train_dl):
            mb_train_input_ids = train_mb["input_ids"].to(device="cuda")
            mb_train_output_ids = train_mb["labels"].to(device="cuda")
            mb_train_loss = 0

            with torch.autocast("cuda", dtype=torch.bfloat16):
                mb_train_outputs = model(input_ids=mb_train_input_ids, labels=mb_train_output_ids, use_cache=False)

            if args.gradient_accumulation_steps > 1:
                mb_train_outputs.loss /= args.gradient_accumulation_steps
                mb_train_outputs.loss.backward()
                mb_train_loss += mb_train_outputs.loss.item()
                train_steps += 1 / args.gradient_accumulation_steps

                if (mb_train_index + 1) % args.gradient_accumulation_steps == 0:
                    clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    pbar.update()
            else:
                train_steps += 1
                mb_train_outputs.loss.backward()
                mb_train_loss += mb_train_outputs.loss.item()
                clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                pbar.update()

            if args.torch_empty_cache_every_steps:
                del mb_train_input_ids, mb_train_output_ids, mb_train_outputs
                torch.cuda.empty_cache()

            if train_steps % args.logging_steps == 0:
                valid_dl = DataLoader(
                    valid_ds,
                    batch_size=args.per_device_eval_batch_size,
                    drop_last=False,
                    shuffle=False,
                    collate_fn=collate_fn,
                )
                valid_loss = 0
                model.eval()

                for valid_step, valid_mb in enumerate(valid_dl):
                    mb_valid_input_ids = valid_mb["input_ids"].cuda()
                    mb_valid_output_ids = valid_mb["labels"].cuda()

                    with torch.no_grad():
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            mb_valid_outputs = model(
                                input_ids=mb_valid_input_ids,
                                labels=mb_valid_output_ids,
                                use_cache=False,
                            )
                    valid_loss += mb_valid_outputs.loss.item()

                    if (valid_step + 1) == args.eval_steps:
                        valid_loss /= valid_step + 1
                        break

                model.train()
                pbar.write(f"steps: {int(train_steps)}, train_loss: {mb_train_loss:.4f}, valid_loss: {valid_loss:.4f}")
                del mb_valid_input_ids, mb_valid_output_ids, mb_valid_outputs
                torch.cuda.empty_cache()

            if train_steps % args.save_steps == 0:
                model.save_pretrained(f"{args.save_dirpath}/{int(train_steps):09d}")
                tokenizer.save_pretrained(f"{args.save_dirpath}/{int(train_steps):09d}")

            if train_steps == args.training_steps:
                break


if __name__ == "__main__":
    main()
