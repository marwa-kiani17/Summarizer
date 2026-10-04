# News Article Summarizer

Fine-tune a small language model ([Qwen3-4B-Instruct-2507](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507)) to write short news summaries, using QLoRA on the [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset. Everything runs on a single 12 GB NVIDIA GPU.

## Results

Scored on the same 200 randomly chosen CNN/DailyMail test articles (greedy decoding, up to 200 new tokens):

| Metric | Base model | Fine-tuned | Change |
|---|---|---|---|
| ROUGE-1 | 34.03 | **39.89** | +5.86 |
| ROUGE-2 | 10.65 | **17.19** | +6.54 |
| ROUGE-L | 20.37 | **27.14** | +6.77 |
| ROUGE-Lsum | 30.33 | **37.43** | +7.10 |

Validation loss went from 1.363 (step 100) to 1.293 (step 625). With 200 articles, scores can move by about 1-2 points between samples, but the gap between the two models is much larger than that. See `Training_Report.pdf` for charts and the full setup.

Example output (unseen article, "United States announces blockade on the Strait of Hormuz"):

```
President Donald Trump announces blockade on the Strait of Hormuz .
Blockade follows collapse of talks between the U.S. and Iran .
Oil prices rise above US$100 a barrel, then ease back to just over US$99 .
```

The space before each full stop is the CNN/DailyMail dataset's own style.

## Requirements

- NVIDIA GPU with 12 GB of memory (tested on an RTX 3060)
- Python 3.12 (3.10-3.12 work; 3.14 does not)
- About 30 GB of free disk space
- Windows (native) or Linux / WSL2

## Setup

Native Windows with Conda (what this project was trained with):

```bat
conda create -n summarizer python=3.12
conda activate summarizer
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Check that PyTorch sees your GPU. This must print `True` and your GPU name:

```bat
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

On Linux or WSL2 the steps are the same; a normal `python -m venv .venv` also works instead of Conda. Install PyTorch first, then `requirements.txt`.

## Usage

### 1. Quick test (about 15 minutes, mostly downloads on the first run)

```bat
python train.py --max_steps 30 --train_size 500
```

### 2. Train

```bat
python train.py
```

10,000 articles, 1 epoch, 625 steps. Expect roughly 4 hours on an RTX 3060. The loss is printed every 10 steps and should settle around 1.3-1.4. A checkpoint is saved every 25 steps, so if training stops, continue with:

```bat
python train.py --resume
```

The final adapter is saved to `outputs/qwen3-4b-cnndm-lora`.

If you run out of GPU memory: `python train.py --batch_size 1 --grad_accum 16`, or lower `MAX_SEQ_LENGTH` to 1536 in `common.py`.

### 3. Summarize

```bat
python summarize.py --sample                  :: random CNN/DailyMail test article, with the real summary
python summarize.py --file my_article.txt     :: your own text file (UTF-8, English news works best)
python summarize.py --sample --base           :: same, with the untuned base model
```

Articles longer than about 1,800 tokens (roughly 1,300 words) are cut off, and only the start is summarized.

### 4. Score with ROUGE

```bat
python evaluate_rouge.py --model base --n 200 --batch_size 4
python evaluate_rouge.py --model finetuned --n 200 --batch_size 4
```

Use the same `--n` for both runs so the comparison is fair. `--batch_size 4` keeps GPU memory from filling up on a 12 GB card; the default of 8 can slow down a lot. Scores and every generated summary are saved to `outputs/eval_base.json` and `outputs/eval_finetuned.json`.

### 5. Export (optional)

```bat
python export.py --format merged     :: standalone 16-bit model (about 8 GB)
python export.py --format gguf       :: 4-bit GGUF for llama.cpp / Ollama (about 2.5 GB)
```

GGUF export is more reliable on Linux / WSL2 than on native Windows.

## Project files

| File | What it does |
|---|---|
| `common.py` | Shared settings, prompt format, and output cleanup |
| `train.py` | QLoRA fine-tuning |
| `summarize.py` | Summarize a sample or your own article |
| `evaluate_rouge.py` | ROUGE scoring, base vs fine-tuned |
| `export.py` | Merge the adapter or export to GGUF |
| `requirements.txt` | Python packages |
| `Training_Report.pdf` | Charts and stats for the training run |

## Training setup

| | |
|---|---|
| Base model | unsloth/Qwen3-4B-Instruct-2507, 4-bit |
| Method | QLoRA, rank 16, all attention and MLP layers |
| Trainable parameters | 33,030,144 of 4,055,498,240 (0.81%) |
| Data | 10,000 training / 500 validation articles (over 2,048 tokens removed) |
| Batch | 2 per step x 8 accumulation = 16 |
| Learning rate | 2e-4, linear decay, 30 warm-up steps |
| Optimizer / precision | AdamW 8-bit / bfloat16 |
| Loss | Computed on the summary only, not on the article |

## Notes

- The prompt format in `common.py` must stay identical between training and inference. If you change it, retrain.
- The model is trained on English news. Other text types and languages give weaker results.
- ROUGE measures word overlap with the reference summaries, not truthfulness. Read some summaries yourself.
- Given an empty input, a language model invents a plausible-sounding story. `summarize.py` stops with an error if the file is empty.
- Qwen3's chat template adds an empty `<think></think>` block to training examples, so the model writes one before each summary. `common.clean_output` removes it.
- The trained model is not included in this repository because the weights file is over GitHub's 100 MB limit.

## Credits

Built with [Unsloth](https://github.com/unslothai/unsloth), [Hugging Face Transformers and TRL](https://github.com/huggingface), the [Qwen3](https://huggingface.co/Qwen) model, and the [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset. Check each project's license before reusing the model or data.
