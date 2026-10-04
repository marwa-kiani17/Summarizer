"""Summarize an article with the fine-tuned (or base) model.

Usage:
    python summarize.py --sample                 # random CNN/DM test article
    python summarize.py --file my_article.txt    # your own text file
    python summarize.py --sample --base          # same, with the untuned base model
"""
import argparse
import random

from unsloth import FastLanguageModel
from datasets import load_dataset

import common as C


def load_model(use_base: bool):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=C.BASE_MODEL if use_base else C.ADAPTER_DIR,
        max_seq_length=C.MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)  # enables faster generation
    return model, tokenizer


def summarize(model, tokenizer, article: str, temperature: float = 0.0,
              max_new_tokens: int = 200, repetition_penalty: float = 1.1) -> str:
    article = C.truncate_article(tokenizer, article)
    prompt = C.build_prompt(tokenizer, article)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens,
                      repetition_penalty=repetition_penalty,
                      pad_token_id=tokenizer.eos_token_id)
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)
    out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return C.clean_output(tokenizer.decode(new_tokens, skip_special_tokens=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=str, help="Path to a .txt file to summarize")
    p.add_argument("--sample", action="store_true", help="Use a random test article")
    p.add_argument("--base", action="store_true", help="Use the untuned base model")
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()

    if not args.file and not args.sample:
        p.error("give --file PATH or --sample")

    model, tokenizer = load_model(args.base)
    reference = None
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            article = f.read()
    else:
        test = load_dataset(C.DATASET_NAME, C.DATASET_VERSION, split="test")
        ex = test[random.randrange(len(test))]
        article, reference = ex["article"], ex["highlights"]

    if not article.strip():
        p.error(f"{args.file} is empty - paste your article into it and save the file first")

    print("===== ARTICLE (first 800 chars) =====")
    print(article[:800], "...\n")
    print(f"===== SUMMARY ({'base' if args.base else 'fine-tuned'}) =====")
    print(summarize(model, tokenizer, article, temperature=args.temperature))
    if reference:
        print("\n===== REFERENCE =====")
        print(reference)


if __name__ == "__main__":
    main()
