"""Merge the LoRA adapter into a standalone model, or export to GGUF.

Usage:
    python export.py --format merged   # 16-bit model for vLLM / Transformers (~8GB)
    python export.py --format gguf     # 4-bit GGUF for llama.cpp / Ollama (~2.5GB)
"""
import argparse

from unsloth import FastLanguageModel

import common as C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--format", choices=["merged", "gguf"], required=True)
    p.add_argument("--quant", default="q4_k_m", help="GGUF quantization method")
    args = p.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=C.ADAPTER_DIR,
        max_seq_length=C.MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    if args.format == "merged":
        out = f"{C.OUTPUT_DIR}/merged-16bit"
        model.save_pretrained_merged(out, tokenizer, save_method="merged_16bit")
    else:
        out = f"{C.OUTPUT_DIR}/gguf"
        model.save_pretrained_gguf(out, tokenizer, quantization_method=args.quant)

    print(f"Exported to {out}")


if __name__ == "__main__":
    main()
