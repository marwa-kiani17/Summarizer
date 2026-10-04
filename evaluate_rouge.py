"""Evaluate base vs fine-tuned model with ROUGE on the CNN/DailyMail test set.

Usage:
    python evaluate_rouge.py --model finetuned
    python evaluate_rouge.py --model base
    python evaluate_rouge.py --model finetuned --n 100   # quicker check
"""
import argparse
import json
import os

from unsloth import FastLanguageModel
from datasets import load_dataset
import evaluate
import nltk
from tqdm import tqdm

import common as C


def to_lines(text: str) -> str:
    """ROUGE-Lsum expects one sentence per line."""
    return "\n".join(nltk.sent_tokenize(text.strip()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["finetuned", "base"], default="finetuned")
    p.add_argument("--n", type=int, default=C.TEST_SIZE)
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()

    for pkg in ("punkt", "punkt_tab"):
        nltk.download(pkg, quiet=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=C.BASE_MODEL if args.model == "base" else C.ADAPTER_DIR,
        max_seq_length=C.MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"  # required for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test = load_dataset(C.DATASET_NAME, C.DATASET_VERSION, split="test")
    test = test.shuffle(seed=C.SEED).select(range(min(args.n, len(test))))

    predictions, references, records = [], [], []
    for i in tqdm(range(0, len(test), args.batch_size), desc=f"Generating ({args.model})"):
        batch = test[i:i + args.batch_size]
        prompts = [C.build_prompt(tokenizer, C.truncate_article(tokenizer, a))
                   for a in batch["article"]]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                             repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id)
        gen = out[:, inputs["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for art_id, pred, ref in zip(batch["id"], texts, batch["highlights"]):
            pred = C.clean_output(pred)
            predictions.append(to_lines(pred))
            references.append(to_lines(ref))
            records.append({"id": art_id, "prediction": pred, "reference": ref})

    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    scores = {k: round(v * 100, 2) for k, v in scores.items()}

    print(f"\n===== ROUGE ({args.model}, n={len(test)}) =====")
    for k in ("rouge1", "rouge2", "rougeL", "rougeLsum"):
        print(f"{k:10s} {scores[k]:6.2f}")

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    path = f"{C.OUTPUT_DIR}/eval_{args.model}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"scores": scores, "n": len(test), "samples": records}, f, indent=2)
    print(f"Saved scores and all generated summaries to {path}")


if __name__ == "__main__":
    main()
