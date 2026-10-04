"""Fine-tune Qwen3-4B-Instruct-2507 on CNN/DailyMail with QLoRA (12GB GPU).

Usage:
    python train.py                                  # full run
    python train.py --max_steps 30 --train_size 500  # quick smoke test
    python train.py --batch_size 1 --grad_accum 16   # if you hit OOM
    python train.py --resume                         # continue from last checkpoint
"""
import argparse

from unsloth import FastLanguageModel, is_bfloat16_supported  # import unsloth first
from unsloth.chat_templates import train_on_responses_only

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

import common as C


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_size", type=int, default=C.TRAIN_SIZE)
    p.add_argument("--val_size", type=int, default=C.VAL_SIZE)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1, help="-1 = use epochs")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def prepare_split(tokenizer, split: str, n: int):
    """Load a split, format as chat text, drop examples longer than MAX_SEQ_LENGTH."""
    ds = load_dataset(C.DATASET_NAME, C.DATASET_VERSION, split=split)
    # Take a bit extra so we still have n examples after length filtering
    ds = ds.shuffle(seed=C.SEED).select(range(min(len(ds), int(n * 1.15))))

    def to_text(ex):
        text = tokenizer.apply_chat_template(
            C.build_messages(ex["article"], ex["highlights"]), tokenize=False
        )
        n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        return {"text": text, "n_tokens": n_tokens}

    ds = ds.map(to_text, remove_columns=ds.column_names, num_proc=2)
    before = len(ds)
    ds = ds.filter(lambda ex: ex["n_tokens"] <= C.MAX_SEQ_LENGTH)
    print(f"[{split}] kept {len(ds)}/{before} examples within {C.MAX_SEQ_LENGTH} tokens")
    return ds.select(range(min(n, len(ds)))).remove_columns(["n_tokens"])


def main():
    args = parse_args()

    # ---- Step 3: model in 4-bit + LoRA adapters ----
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=C.BASE_MODEL,
        max_seq_length=C.MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,  # auto: bf16 on supported GPUs, else fp16
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=C.SEED,
    )
    model.print_trainable_parameters()

    # ---- Step 2: data ----
    train_ds = prepare_split(tokenizer, "train", args.train_size)
    val_ds = prepare_split(tokenizer, "validation", args.val_size)
    print("\n===== Sample training text =====")
    print(train_ds[0]["text"][:1500], "...\n")

    # ---- Step 4: training ----
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            warmup_steps=30,
            lr_scheduler_type="linear",
            optim="adamw_8bit",
            weight_decay=0.01,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=25,
            save_total_limit=2,
            output_dir=C.CHECKPOINT_DIR,
            seed=C.SEED,
            report_to="none",
        ),
    )

    # Only compute loss on the assistant's summary, not on the prompt/article
    trainer = train_on_responses_only(
        trainer,
        instruction_part=C.INSTRUCTION_PART,
        response_part=C.RESPONSE_PART,
    )

    stats = trainer.train(resume_from_checkpoint=True if args.resume else None)
    print(f"\nTraining done in {stats.metrics['train_runtime'] / 60:.1f} min")

    model.save_pretrained(C.ADAPTER_DIR)
    tokenizer.save_pretrained(C.ADAPTER_DIR)
    print(f"LoRA adapter saved to {C.ADAPTER_DIR}")


if __name__ == "__main__":
    main()
