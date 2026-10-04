"""Shared settings and prompt format used by every script."""
import re

# ---------------- Model & data settings ----------------
BASE_MODEL = "unsloth/Qwen3-4B-Instruct-2507"
DATASET_NAME = "abisee/cnn_dailymail"
DATASET_VERSION = "3.0.0"

MAX_SEQ_LENGTH = 2048     # article + summary tokens; lower to 1536 if you hit OOM
TRAIN_SIZE = 10_000
VAL_SIZE = 500
TEST_SIZE = 500
SEED = 3407

OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"
ADAPTER_DIR = f"{OUTPUT_DIR}/qwen3-4b-cnndm-lora"

# ---------------- Prompt format ----------------
# Keep this identical between training and inference. If you change it, retrain.
SYSTEM_PROMPT = "You are a helpful assistant that writes concise, factual news summaries."
USER_INSTRUCTION = "Summarize the following news article in 3-4 short highlight sentences."

# Chat template markers for Qwen3 (used to mask loss to the assistant reply only)
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"


def build_messages(article: str, summary: str | None = None) -> list[dict]:
    """Turn an article (and optional target summary) into chat messages."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{USER_INSTRUCTION}\n\nArticle:\n{article.strip()}"},
    ]
    if summary is not None:
        messages.append({"role": "assistant", "content": summary.strip()})
    return messages


def build_prompt(tokenizer, article: str) -> str:
    """Prompt text ready for generation (ends where the assistant reply starts)."""
    return tokenizer.apply_chat_template(
        build_messages(article), tokenize=False, add_generation_prompt=True
    )


def clean_output(text: str) -> str:
    """Remove the empty <think></think> block Qwen3's template adds before the reply."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def truncate_article(tokenizer, article: str, max_tokens: int = 1800) -> str:
    """Cut an article to max_tokens so prompt + summary fit in the context."""
    ids = tokenizer(article, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return article
    return tokenizer.decode(ids[:max_tokens])
