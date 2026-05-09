import json

import pytest

from src.train_sft import (
    SFTRecord,
    format_prompt_prefix,
    format_training_text,
    load_config,
    load_jsonl_records,
    tokenize_record,
)


class FakeTokenizer:
    chat_template = None

    def __call__(
        self,
        text,
        truncation=False,
        max_length=None,
        padding=False,
    ):
        input_ids = [ord(char) for char in text]
        if truncation and max_length is not None:
            input_ids = input_ids[:max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }


class FakeChatTokenizer(FakeTokenizer):
    chat_template = "fake-template"

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
    ):
        parts = []
        for message in messages:
            parts.append(f"<{message['role']}>{message['content']}</{message['role']}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)


def test_load_config_reads_expected_sft_config():
    config = load_config("configs/train_sft_deon.yaml")

    assert config["model_name"] == "google/gemma-2-2b-it"
    assert config["train_path"] == "data/train/deon_sft_train.jsonl"
    assert config["val_path"] == "data/train/deon_sft_val.jsonl"
    assert config["output_dir"] == "checkpoints/gemma2_deon_sft_final"
    assert config["max_seq_length"] == 1024
    assert config["learning_rate"] == pytest.approx(2.0e-4)
    assert config["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_load_config_rejects_missing_required_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("model_name: google/gemma-2-2b-it\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing keys"):
        load_config(path)


def test_load_jsonl_records_reads_prompt_completion_pairs(tmp_path):
    path = tmp_path / "records.jsonl"
    records = [
        {"prompt": "Prompt one", "completion": "action1", "extra": "ignored"},
        {"prompt": "Prompt two", "completion": "action1"},
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    loaded = load_jsonl_records(path)

    assert loaded == [
        SFTRecord(prompt="Prompt one", completion="action1"),
        SFTRecord(prompt="Prompt two", completion="action1"),
    ]


def test_load_jsonl_records_rejects_missing_prompt_or_completion(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({"prompt": "Prompt only"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="completion"):
        load_jsonl_records(path)


def test_generated_train_and_val_records_have_prompt_completion():
    for path in [
        "data/train/deon_sft_train.jsonl",
        "data/train/deon_sft_val.jsonl",
    ]:
        records = load_jsonl_records(path)

        assert records
        assert all(record.prompt for record in records)
        assert all(record.completion for record in records)


def test_format_training_text_fallback_contains_prompt_and_completion():
    record = SFTRecord(prompt="Choose one.", completion="action1")

    text = format_training_text(record, FakeTokenizer())

    assert "Choose one." in text
    assert "action1" in text
    assert text == "Choose one.\naction1"


def test_format_training_text_chat_template_contains_prompt_and_completion():
    record = SFTRecord(prompt="Choose one.", completion="action1")

    text = format_training_text(record, FakeChatTokenizer())

    assert "<user>Choose one.</user>" in text
    assert "<assistant>action1</assistant>" in text


def test_format_prompt_prefix_uses_generation_prompt_for_chat_template():
    record = SFTRecord(prompt="Choose one.", completion="action1")

    text = format_prompt_prefix(record, FakeChatTokenizer())

    assert text == "<user>Choose one.</user><assistant>"


def test_tokenize_record_masks_prompt_prefix_labels():
    record = SFTRecord(prompt="Prompt", completion="action1")

    tokenized = tokenize_record(record, FakeTokenizer(), max_seq_length=128)

    prompt_length = len("Prompt\n")
    assert tokenized["labels"][:prompt_length] == [-100] * prompt_length
    assert tokenized["labels"][prompt_length:] == tokenized["input_ids"][prompt_length:]
    assert len(tokenized["input_ids"]) == len(tokenized["attention_mask"])
    assert len(tokenized["input_ids"]) == len(tokenized["labels"])
