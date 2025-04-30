# tests/dataloader
## preliminary
```bash
python download_wiki.py --help
usage: download_wiki.py [-h] [--lang {ko,en}] [--num_proc NUM_PROC]

options:
  -h, --help           show this help message and exit
  --lang {ko,en}
  --num_proc NUM_PROC
```
```bash
datasets/
└── wiki
    ├── en
    │   ├── 000000.jsonl
    │   ├── 000001.jsonl
    │   ├── 000002.jsonl
    │   ├── ...
    │   ├── 000045.jsonl
    │   ├── 000046.jsonl
    │   └── 000047.jsonl
    └── ko
        ├── 000000.jsonl
        ├── 000001.jsonl
        ├── 000002.jsonl
        └── 000003.jsonl
```