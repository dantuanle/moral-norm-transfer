"""QLoRA supervised fine-tuning for the deontological IPD SFT data."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REQUIRED_CONFIG_KEYS = {
    "model_name",
    "train_path",
    "val_path",
    "output_dir",
    "max_seq_length",
    "learning_rate",
    "num_train_epochs",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "seed",
}


@dataclass(frozen=True)
class SFTRecord:
    prompt: str
    completion: str


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and lightly validate a YAML training config."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping in {config_path}")

    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"Config {config_path} is missing keys: {missing}")

    if not isinstance(config["target_modules"], list) or not all(
        isinstance(module, str) for module in config["target_modules"]
    ):
        raise ValueError("target_modules must be a list of strings")

    return config


def load_jsonl_records(path: str | Path) -> list[SFTRecord]:
    """Load prompt/completion pairs from generated SFT JSONL."""
    records: list[SFTRecord] = []
    jsonl_path = Path(path)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            raw_record = json.loads(line)
            prompt = raw_record.get("prompt")
            completion = raw_record.get("completion")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{jsonl_path}:{line_number} missing non-empty prompt")
            if not isinstance(completion, str) or not completion.strip():
                raise ValueError(
                    f"{jsonl_path}:{line_number} missing non-empty completion"
                )
            records.append(SFTRecord(prompt=prompt, completion=completion))

    return records


def _has_chat_template(tokenizer: Any) -> bool:
    return callable(getattr(tokenizer, "apply_chat_template", None)) and bool(
        getattr(tokenizer, "chat_template", None)
    )


def format_training_text(record: SFTRecord, tokenizer: Any) -> str:
    """Format one record as user prompt plus assistant completion."""
    if _has_chat_template(tokenizer):
        messages = [
            {"role": "user", "content": record.prompt},
            {"role": "assistant", "content": record.completion},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    return f"{record.prompt}\n{record.completion}"


def format_prompt_prefix(record: SFTRecord, tokenizer: Any) -> str:
    """Format the prompt-only prefix used for label masking."""
    if _has_chat_template(tokenizer):
        messages = [{"role": "user", "content": record.prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"{record.prompt}\n"


def tokenize_record(
    record: SFTRecord,
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, list[int]]:
    """Tokenize and mask labels so only completion tokens contribute loss."""
    text = format_training_text(record, tokenizer)
    prompt_prefix = format_prompt_prefix(record, tokenizer)

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )
    prefix_tokenized = tokenizer(
        prompt_prefix,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )

    input_ids = list(tokenized["input_ids"])
    attention_mask = list(tokenized["attention_mask"])
    prompt_length = min(len(prefix_tokenized["input_ids"]), len(input_ids))
    labels = [-100] * prompt_length + input_ids[prompt_length:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class SFTDataset:
    """Tiny list-backed dataset for Hugging Face Trainer."""

    def __init__(
        self,
        records: list[SFTRecord],
        tokenizer: Any,
        max_seq_length: int,
    ) -> None:
        self.examples = [
            tokenize_record(record, tokenizer, max_seq_length) for record in records
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def train(config: dict[str, Any]) -> None:
    """Run QLoRA SFT and save the resulting adapter/tokenizer."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(int(config["seed"]))

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_records = load_jsonl_records(config["train_path"])
    val_records = load_jsonl_records(config["val_path"])
    train_dataset = SFTDataset(
        train_records,
        tokenizer,
        max_seq_length=int(config["max_seq_length"]),
    )
    val_dataset = SFTDataset(
        val_records,
        tokenizer,
        max_seq_length=int(config["max_seq_length"]),
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=quantization_config,
        device_map="auto",
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=list(config["target_modules"]),
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=float(config["num_train_epochs"]),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=int(config["seed"]),
        remove_unused_columns=False,
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
