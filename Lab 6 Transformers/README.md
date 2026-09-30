# Lab 6: Learning AR Models (Transformers)

Hands-on session from `labs/transformers/`: the three Transformer families, and a GPT-type model run locally with Ollama and compared with a frontier model.

## Files

- `transformers_lab.ipynb` - the executed notebook with outputs and observations after every section.

## What is in it

1. Encoder-decoder translation: `google/bert2bert_L-24_wmt_en_de` (EN to DE), `google-t5/t5-base` (EN to FR/DE, summarisation by prefix), `rvv-karma/English2Hinglish-Flan-T5-Base`.
2. Encoder-only: DistilBERT sentiment analysis (including where it fails), plus semantic search and clustering with `all-MiniLM-L6-v2` embeddings.
3. Decoder-only: GPT-2 generation, the full next-token distribution, greedy vs sampling, and a numerical check of causal masking.
4. Ollama: `llama3.2:3b` through LangChain (as in `Run_Ollama.ipynb`), with answers compared against Claude.

## Running it

Needs `transformers`, `torch`, `sentencepiece`, `langchain-ollama`, a running Ollama with `llama3.2:3b` pulled, and about 5.5 GB of model downloads on first run.

```bash
ollama pull llama3.2:3b
jupyter nbconvert --to notebook --execute transformers_lab.ipynb --output transformers_lab.ipynb
```
