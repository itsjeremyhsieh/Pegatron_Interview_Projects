## A2: MMMU MCQ Benchmark (OpenAI Vision)

This script evaluates a vision-language model on MMMU multiple-choice questions.

### Features
- Loads MMMU split by subject (e.g., Accounting, Computer_Science)
- Sends question + image + MCQ options to an OpenAI vision-capable model
- Forces single-letter answers (A/B/C/...) and computes accuracy

### Structure
- `A2/data_loader.py` – loads and preprocesses MMMU (image_1, options, label)
- `A2/model_interface.py` – OpenAI client; formats prompt and image
- `A2/evaluator.py` – runs the loop, normalizes model output to a letter
- `A2/run_benchmark.py` – CLI entry point

### Setup
1) Python 3.10+
2) Install deps:
```
pip install -r A2/requirements.txt
```
1) Set your OpenAI API key (zsh/bash):
```
export OPENAI_API_KEY="sk-..."
```

### Run
```
python3 A2/run_benchmark.py \
  --model gpt-4o-mini \
  --subject Accounting \
  --max_samples 10
```

Flags:
- `--model` – OpenAI model name (vision-capable), e.g. `gpt-4o-mini`, `gpt-4o`
- `--subject` – MMMU subject subset (e.g., `Accounting`, `Computer_Science`)
- `--max_samples` – limit evaluated samples

### Notes
- The evaluator trusts the API to return a single letter; a small regex extracts A–J.
- Images are passed as data URLs (JPEG) to the OpenAI chat completions API.
- If you observe non-letter outputs, tighten the prompt in `A2/model_interface.py`.

### Troubleshooting
- "Debug - OpenAI client not initialized" → ensure `openai` installed and `OPENAI_API_KEY` set
- Model responding with unexpected text → try `gpt-4o` and reduce `max_samples` first


