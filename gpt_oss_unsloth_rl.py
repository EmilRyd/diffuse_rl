from unsloth import FastLanguageModel
from transformers import PreTrainedTokenizer
from trl import GRPOConfig, GRPOTrainer
import random
from dotenv import load_dotenv
from datasets import Dataset
import pandas as pd
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any
from openai import OpenAI
import os
import re

load_dotenv()
from templates import AQUARAT_TEMPLATE_STYLIZED_RED_TEAM, DEFAULT_GT_INSTRUCTIONS, DEFAULT_GT_TEMPLATE


@dataclass(frozen=True, slots=True)
class GPTOssGRPOConfig:
    model_name: str
    lora_rank: int
    reasoning_effort: str
    filter_out_prompts_that_are_too_long: bool
    training_args: GRPOConfig


@dataclass(frozen=True, slots=True)
class Datapoint:
    prompt: list[dict[str, str]]
    extra_data: Any = None


def grpo_train(
    dataset: list[Datapoint],
    reward_function: Callable[[str, Any], float],
    cfg: GPTOssGRPOConfig,
) -> None:
    model, tokenizer = get_model_and_tokenizer(cfg)

    dataset = dataset.copy()
    random.Random(42).shuffle(dataset)

    dataset = filter_too_long(dataset=dataset, tokenizer=tokenizer, cfg=cfg)

    trainer = get_grpo_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        reward_function=reward_function,
        cfg=cfg,
    )

    trainer.train()


def filter_too_long(
    dataset: list[Datapoint], tokenizer: PreTrainedTokenizer, cfg: GPTOssGRPOConfig
) -> list[Datapoint]:
    filtered_dataset: list[Datapoint] = [
        datapoint
        for datapoint in dataset
        if len(tokenizer.apply_chat_template(datapoint.prompt))
        <= cfg.training_args.max_prompt_length - 8  # -8 just in case
    ]

    n_filtered_out: int = len(dataset) - len(filtered_dataset)

    if cfg.filter_out_prompts_that_are_too_long:
        assert n_filtered_out == 0, (
            "Some prompts are longer than cfg.training_args.max_prompt_length. Please increase cfg.training_args.max_prompt_length, make the prompts shorter, or pass filter_out_prompts_that_are_too_long=True in GPTOssGRPOConfig to filter out those prompts"
        )

    if n_filtered_out > 0:
        print(
            f"WARNING: Filtered out {n_filtered_out} out of {len(dataset)} (or {n_filtered_out / len(dataset):.2%}) datapoints because their prompts were longer than cfg.training_args.max_prompt_length."
        )

    assert len(filtered_dataset) > 0, (
        "You passed a dataset or passed a dataset all of whose prompts are longer than cfg.training_args.max_prompt_length"
    )

    return filtered_dataset


def get_model_and_tokenizer(
    cfg: GPTOssGRPOConfig,
) -> tuple[FastLanguageModel, PreTrainedTokenizer]:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.training_args.max_prompt_length
        + cfg.training_args.max_completion_length,
        load_in_4bit=True,  # False for LoRA 16bit
        offload_embedding=True,  # Reduces VRAM by 1GB
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=cfg.lora_rank * 2,  # *2 speeds up training
        use_gradient_checkpointing="unsloth",  # Reduces memory usage
        random_state=3407,
    )

    return model, tokenizer


def get_grpo_trainer(
    model: FastLanguageModel,
    tokenizer: PreTrainedTokenizer,
    dataset: list[Datapoint],
    reward_function: Callable[[str, Any], float],
    cfg: GPTOssGRPOConfig,
) -> GRPOTrainer:
    def wrapped_reward_function(
        completions: list[dict[str, str]], extra_data: list[Any], **kwargs
    ) -> list[float]:
        print(f"Computing {len(completions)} rewards.")
        rewards: list[float] = []

        for completion, extra_datum in zip(completions, extra_data, strict=True):
            assert isinstance(completion, list)
            assert len(completion) == 1
            assert isinstance(completion[0], dict)
            assert set(completion[0].keys()) == {"role", "content"}
            assert completion[0]["role"] == "assistant"
            assert isinstance(completion[0]["content"], str)

            reward: float = reward_function(completion[0]["content"], extra_datum)

            if isinstance(reward, int):
                reward = float(reward)
            assert isinstance(reward, float)

            rewards.append(reward)

        return rewards

    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[wrapped_reward_function],
        args=cfg.training_args,
        train_dataset=Dataset.from_list(
            [
                {
                    "prompt": datapoint.prompt,
                    "extra_data": datapoint.extra_data,
                    "reasoning_effort": cfg.reasoning_effort,
                }
                for datapoint in dataset
            ]
        ),
    )


def my_reward_function(completion: str, extra_data: Any) -> float:
    expected_output: int = extra_data["expected_output"]
    assert isinstance(expected_output, int)

    got_output_str: str = completion.split("\\boxed{")[-1].split("}")[0]
    try:
        got_output: int = int(got_output_str)
    except ValueError:
        return 0.0

    return -min(1.0, ((expected_output - got_output) / 128) ** 2)

def ground_truth_reward_function(completion: str, extra_data: Any) -> float:
    true_answer = extra_data['stored_incorrect_answer']#extra_data['true_answer']
    question = extra_data['question']

    print(completion)

    reward_model_prompt = DEFAULT_GT_TEMPLATE.format(question=question, answer=completion, criterion=true_answer, instructions=DEFAULT_GT_INSTRUCTIONS)

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": reward_model_prompt}]
    )

    openai_output = response.choices[0].message.content

    print(openai_output)
    # parse the response
    # extract the grade
    match = None
    for m in re.finditer(r"GRADE:(.*)", openai_output):
        match = m
    if match:
        value = match.group(1)
        value = ''.join(filter(str.isdigit, value)) # remove anything that's not a digit
    else:
        # if no grade is found, return 0 and log a warning
        # try look for 'Grade: '
        match = None
        for m in re.finditer(r"Grade:(.*)", openai_output):
            match = m
        if match:
            value = match.group(1)
            value = ''.join(filter(str.isdigit, value)) # remove anything that's not a digit
        else:
            value = ''
    
    if value is None or value == '':
        return None
    else:
        value = float(value)
        if value > 10.0:
            print(f"Value is greater than 10.0: {value}, clipping to 10.0")
            value = 10.0
        elif value < 0.0:
            print(f"Value is less than 0.0: {value}, clipping to 0.0")
            value = 0.0
    
    return float(value) / 10.0

if __name__ == "__main__":

    def f(x: int) -> int:
        return x + 123

    DATASET_SIZE = 1024

    random_inputs: list[int] = [random.randint(-128, 127) for _ in range(DATASET_SIZE)]

    '''dataset: list[Datapoint] = [
        Datapoint(
            prompt=[
                {
                    "role": "user",
                    "content": f"What is f{x}? Please put the numeric answer in \\boxed{{ ... }}.",
                }
            ],
            extra_data={"expected_output": f(x)},
        )
        for x in random_inputs
    ]'''


    # Load the dataset from HuggingFace
    olympiads_dataset = pd.read_csv("data/olympiads.csv")

    # For demonstration, let's use the 'train' split and convert it to our Datapoint format
    dataset: list[Datapoint] = [
        Datapoint(
            prompt=[
                {
                    "role": "user",
                    "content": AQUARAT_TEMPLATE_STYLIZED_RED_TEAM.format(incorrect_answer=example["stored_incorrect_answer"]),
                }
            ],
            extra_data={'stored_incorrect_answer': example.get("stored_incorrect_answer", None), 'true_answer': example.get("target", None), 'question': example.get("question", None)}
        )
        for example in olympiads_dataset.to_dict('records')
    ]

    cfg = GPTOssGRPOConfig(
        model_name="unsloth/gpt-oss-20b",
        lora_rank=16,
        reasoning_effort="low",
        filter_out_prompts_that_are_too_long=False,
        training_args=GRPOConfig(
            temperature=1.0,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            optim="adamw_8bit",
            logging_steps=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_generations=4,
            # per_device_train_batch_size=32,
            # generation_batch_size=32,
            # gradient_accumulation_steps=32,
            # num_generations=4,
            max_prompt_length=1024,
            max_completion_length=2048,
            num_train_epochs=1,
            # max_steps=100,
            # save_steps=100,
            report_to="wandb",  # Can use Weights & Biases
            output_dir="outputs",
            # For optional training + evaluation
            # fp16_full_eval = True,
            # per_device_eval_batch_size = 4,
            # eval_accumulation_steps = 1,
            # eval_strategy = "steps",
            # eval_steps = 1,
        ),
    )

    grpo_train(dataset=dataset, reward_function=ground_truth_reward_function, cfg=cfg)