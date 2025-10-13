from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Mxfp4Config,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import get_peft_model, LoraConfig, PeftModel
import torch
from torch.optim import Optimizer, AdamW
import random
from dotenv import load_dotenv
from datasets import Dataset
import wget
import pandas as pd
from argparse import ArgumentParser
from itertools import count
from dataclasses import dataclass, field
from os.path import isfile
from collections.abc import Callable, Awaitable
from typing import Any
import asyncio
from openai import AsyncOpenAI
import os
import re

load_dotenv()
from templates import (
    AQUARAT_TEMPLATE_STYLIZED_RED_TEAM,
    DEFAULT_GT_INSTRUCTIONS,
    DEFAULT_GT_TEMPLATE,
)


@dataclass(frozen=True, slots=True)
class GRPOConfig:
    huggingface_model_name: str = "unsloth/gpt-oss-20b-bf16"
    gguf_model_path: str = "weights/gpt-oss-20b-mxfp4.gguf"
    gguf_model_url: str = "https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-mxfp4.gguf"
    lora_rank: int = 16
    learning_rate: int = 1e-3


def load_huggingface_transformer(cfg: GRPOConfig) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(
        cfg.huggingface_model_name,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
    ).cuda()

    model.train()

    peft_config = peft_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=2 * cfg.lora_rank,
        target_modules="all-linear",
    )

    model = get_peft_model(model, peft_config)

    return model


def load_tokenizer(cfg: GRPOConfig) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(cfg.huggingface_model_name)


def download_gguf_model(cfg: GRPOConfig) -> None:
    if isfile(cfg.gguf_model_path):
        return
    wget.download(cfg.gguf_model_url, cfg.gguf_model_path)


def make_optimizer(model: PeftModel, cfg: GRPOConfig) -> Optimizer:
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    return AdamW(trainable_params, lr=cfg.learning_rate)


def main() -> None:
    cfg = GRPOConfig()

    training_model = load_huggingface_transformer(cfg)

    optimizer = make_optimizer(training_model, cfg)

    tokenizer = load_tokenizer(cfg)

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Please say something."}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).cuda()

    datapoint = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Please say something."},
            {"role": "assistant", "content": "................................."},
        ],
        return_tensors="pt",
    ).cuda()

    for _ in range(64):
        optimizer.zero_grad()
        training_model(
            datapoint[:, :-1], labels=datapoint[:, 1:], return_logits=False
        ).loss.backward()
        optimizer.step()

        generated = training_model.generate(prompt)
        print(tokenizer.batch_decode(generated))

    training_model.save_pretrained("lora-adapter")


if __name__ == "__main__":
    main()
