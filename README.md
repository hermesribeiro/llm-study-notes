# LLM Study Notes

## Byte pair encoding tokenization

### Downloading the data

```bash
curl -fLo data/TinyStoriesV2-GPT4-valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -fLo data/TinyStoriesV2-GPT4-valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```

### Training the tokenizer

```bash
uv run train-bpe data/TinyStoriesV2-GPT4-valid.txt
uv run train-bpe data/TinyStoriesV2-GPT4-train.txt
```

Will create the files `data/TinyStoriesV2-GPT4-valid_merges.txt`, `data/TinyStoriesV2-GPT4-valid_vocab.txt`, `data/TinyStoriesV2-GPT4-train_merges.txt` and `data/TinyStoriesV2-GPT4-train_vocab.txt`.
