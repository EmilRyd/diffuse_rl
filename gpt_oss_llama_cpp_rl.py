from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import get_peft_model, LoraConfig, PeftModel
import torch
from torch import Tensor
from torch.optim import Optimizer, AdamW
from dotenv import load_dotenv
from subprocess import run, Popen, DEVNULL
import wget
import pandas as pd
import wandb
from statistics import mean
import aiohttp
from more_itertools import chunked
from itertools import count, chain
from random import Random
from dataclasses import dataclass
from os.path import isfile, join
import tqdm
from collections.abc import Callable, Awaitable
from typing import Any
from jaxtyping import Float, Int
import asyncio
from openai import AsyncOpenAI
import re

load_dotenv()
from templates import (
    AQUARAT_TEMPLATE_STYLIZED_RED_TEAM,
    DEFAULT_GT_INSTRUCTIONS,
    DEFAULT_GT_TEMPLATE,
)


@dataclass(frozen=True, slots=True)
class GSPOConfig:
    huggingface_model_name: str = "unsloth/gpt-oss-20b-bf16"
    gguf_model_path: str = "weights/gpt-oss-20b-mxfp4.gguf"
    gguf_model_url: str = "https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-mxfp4.gguf"
    llama_cpp_server_executable_path: str = "./llama.cpp/build/bin/llama-server"
    llama_cpp_server_first_port: int = 8081
    convert_lora_to_gguf_python_file_path: str = "llama.cpp/convert_lora_to_gguf.py"
    save_adapters_path: str = "adapters"
    lora_rank: int = 16
    learning_rate: int = 1e-4
    max_tokens: int = 2048
    groups_per_epoch: int = 16
    group_size: int = 8
    train_batch_size: int = 32
    clip_epsilon_low: float = 3e-4
    clip_epsilon_high: float = 4e-4
    epochs: int = 1024
    use_wandb: bool = True
    wandb_project: str | None = "gpt-oss-llama-cpp-rl"
    wandb_run_name: str | None = None

    def __post_init__(self) -> None:
        assert (self.groups_per_epoch * self.group_size) % self.train_batch_size == 0


@dataclass(frozen=True, slots=True)
class Datapoint:
    prompt: list[dict[str, str]]
    extra_data: Any = None


def load_huggingface_transformer(cfg: GSPOConfig) -> PreTrainedModel:
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


def load_tokenizer(cfg: GSPOConfig) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(cfg.huggingface_model_name)


def download_gguf_model(cfg: GSPOConfig) -> None:
    if isfile(cfg.gguf_model_path):
        return
    wget.download(cfg.gguf_model_url, cfg.gguf_model_path)


def make_optimizer(model: PeftModel, cfg: GSPOConfig) -> Optimizer:
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    return AdamW(trainable_params, lr=cfg.learning_rate)


def convert_lora_to_gguf(
    huggingface_adapter_path: str, gguf_adapter_filename: str, cfg: GSPOConfig
) -> None:
    run(
        [
            "uv",
            "run",
            cfg.convert_lora_to_gguf_python_file_path,
            "--outfile",
            gguf_adapter_filename,
            huggingface_adapter_path,
        ],
        check=True,
    )


def start_llama_cpp_server(
    gguf_lora_adapter_filename: str | None,
    n_parallel_sequences: int,
    per_sequence_context_length: int,
    port: int,
    cfg: GSPOConfig,
) -> Popen:
    return Popen(
        [
            cfg.llama_cpp_server_executable_path,
            "--model",
            cfg.gguf_model_path,
            "--parallel",
            str(n_parallel_sequences),
            "--ctx-size",
            str(n_parallel_sequences * per_sequence_context_length),
            "--port",
            str(port),
            "--disable-logs",
        ]
        + (
            ["--lora", gguf_lora_adapter_filename]
            if gguf_lora_adapter_filename is not None
            else []
        ),
        stdout=DEVNULL,
        stderr=DEVNULL,
    )


async def generate_single_completion(
    prompt_with_chat_template: list[int], server_port: int, max_tokens: int
) -> str:
    while True:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://localhost:{server_port}/v1/completions",
                json={
                    "prompt": prompt_with_chat_template,
                    "n_predict": max_tokens,
                },
            ) as resp:
                result = await resp.json()

        server_still_loading: bool = (
            "error" in result.keys() and result["error"]["message"] == "Loading model"
        )
        if server_still_loading:
            await asyncio.sleep(1)
            continue

        return result["choices"][0]["text"]


def generate_completions(
    prompts_with_chat_template: list[list[int]],
    gguf_lora_adapter_filename: str | None,
    cfg: GSPOConfig,
) -> list[str]:
    server_port: int = cfg.llama_cpp_server_first_port

    server_process: Popen = start_llama_cpp_server(
        gguf_lora_adapter_filename=gguf_lora_adapter_filename,
        n_parallel_sequences=len(prompts_with_chat_template),
        per_sequence_context_length=cfg.max_tokens,
        port=server_port,
        cfg=cfg,
    )

    async def workload() -> list[str]:
        return await tqdm.asyncio.tqdm.gather(
            *[
                generate_single_completion(
                    prompt,
                    server_port=server_port,
                    max_tokens=cfg.max_tokens,
                )
                for prompt in prompts_with_chat_template
            ],
            desc="generating completions",
        )

    completions: list[str] = asyncio.run(workload())

    server_process.kill()

    return completions


@dataclass(frozen=True, slots=True)
class Rollout:
    prompt: str
    completion: str
    prompt_tokens: list[int]
    completion_tokens: list[int]
    reward: float


def generate_rollouts(
    gguf_lora_adapter_filename: str | None,
    tokenizer: PreTrainedTokenizer,
    data: list[Datapoint],
    reward_function: Callable[[str, Any], Awaitable[float]],
    cfg: GSPOConfig,
) -> list[Rollout]:
    prompts: list[str] = [
        tokenizer.apply_chat_template(
            datapoint.prompt, tokenize=False, add_generation_prompt=True
        )
        for datapoint in data
    ]

    completions: list[str] = generate_completions(
        prompts, gguf_lora_adapter_filename=gguf_lora_adapter_filename, cfg=cfg
    )

    async def compute_rewards() -> list[float]:
        return await tqdm.asyncio.tqdm.gather(
            *[
                reward_function(completion, datapoint.extra_data)
                for completion, datapoint in zip(completions, data, strict=True)
            ],
            desc="computing rewards",
        )

    rewards: list[float] = asyncio.run(compute_rewards())

    rollouts: list[Rollout] = []
    for prompt, completion, reward in zip(prompts, completions, rewards, strict=True):
        prompt_tokens: list[int] = tokenizer(prompt)["input_ids"]
        all_tokens: list[int] = tokenizer(prompt + completion)["input_ids"]
        assert is_prefix(prompt_tokens, all_tokens)
        completion_tokens: list[int] = all_tokens[len(prompt_tokens) :]

        rollouts.append(
            Rollout(
                prompt=prompt,
                completion=completion,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reward=reward,
            )
        )

    return rollouts


def gspo_train(
    cfg: GSPOConfig,
    dataset: list[Datapoint],
    reward_function: Callable[[str, Any], Awaitable[float]],
) -> None:
    model: PeftModel = load_huggingface_transformer(cfg)
    tokenizer: PreTrainedTokenizer = load_tokenizer(cfg)
    optimizer: Optimizer = make_optimizer(model, cfg)

    Random(42).shuffle(dataset)

    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name)

    for epoch in tqdm.tqdm(range(cfg.epochs), desc="gspo training"):
        batch_data: list[Datapoint] = [
            dataset[i % len(dataset)]
            for i in range(
                cfg.groups_per_epoch * epoch, cfg.groups_per_epoch * (epoch + 1)
            )
        ]

        grouped_data: list[list[Datapoint]] = [
            [datapoint] * cfg.group_size for datapoint in batch_data
        ]

        rollouts: list[Rollout] = generate_rollouts(
            gguf_lora_adapter_filename=gguf_lora_adapter_filename(cfg)
            if epoch > 0
            else None,
            tokenizer=tokenizer,
            data=list(chain.from_iterable(grouped_data)),
            reward_function=reward_function,
            cfg=cfg,
        )

        rewards: list[float] = [rollout.reward for rollout in rollouts]

        print("AVERAGE REWARD:", mean(rewards))

        assert len(rewards) % cfg.group_size == 0
        grouped_rewards: list[list[float]] = list(chunked(rewards, cfg.group_size))

        grouped_advantages: list[list[float]] = gspo_advantages(grouped_rewards)
        advantages: list[float] = list(chain.from_iterable(grouped_advantages))

        train(
            model=model,
            optimizer=optimizer,
            rollouts=rollouts,
            advantages=advantages,
            cfg=cfg,
        )

        save_huggingface_lora_adapter(
            model, huggingface_lora_adapter_path(epoch=epoch, cfg=cfg)
        )
        convert_lora_to_gguf(
            huggingface_adapter_path=huggingface_lora_adapter_path(
                epoch=epoch, cfg=cfg
            ),
            gguf_adapter_filename=gguf_lora_adapter_filename(cfg=cfg),
            cfg=cfg,
        )

        if cfg.use_wandb:
            wandb.log({"reward": mean(rewards)})


def huggingface_lora_adapter_path(epoch: int, cfg: GSPOConfig) -> str:
    return join(cfg.save_adapters_path, f"epoch-{epoch}")


def gguf_lora_adapter_filename(cfg: GSPOConfig) -> str:
    return join(cfg.save_adapters_path, "latest-adapter.gguf")


def save_huggingface_lora_adapter(model: PeftModel, path: str) -> None:
    model.save_pretrained(path)


def completion_logprob(model: PeftModel, rollout: Rollout) -> Float[Tensor, ""]:
    tokens: Int[Tensor, " position"] = torch.tensor(
        rollout.prompt_tokens + rollout.completion_tokens, device="cuda"
    )
    in_tokens: Int[Tensor, " position"] = tokens[:-1]
    out_tokens: Int[Tensor, " position"] = tokens[1:]

    all_logits: Float[Tensor, "position vocabulary_size"] = model(
        in_tokens.unsqueeze(0)
    ).logits.squeeze(0)

    all_logprobs = all_logits.log_softmax(-1)

    logprobs: Float[Tensor, " position"] = all_logprobs[
        torch.arange(out_tokens.numel(), device="cuda"), out_tokens
    ]

    return logprobs[-len(rollout.completion_tokens) :].sum()


def gspo_advantages(grouped_rewards: list[list[float]]) -> list[list[float]]:
    grouped_advantages: list[list[float]] = []
    for group in grouped_rewards:
        assert len(group) >= 2
        advantages: list[float] = []
        for reward in group:
            advantages.append(reward - (sum(group) - reward) / (len(group) - 1))
        grouped_advantages.append(advantages)
    return grouped_advantages


def train(
    model: PeftModel,
    optimizer: Optimizer,
    rollouts: list[Rollout],
    advantages: list[float],
    cfg: GSPOConfig,
) -> None:
    with torch.no_grad():
        old_completion_logprobs: list[Float[Tensor, ""]] = [
            completion_logprob(model, rollout)
            for rollout in tqdm.tqdm(rollouts, desc="computing old logprobs")
        ]

    for i, (rollout, advantage, old_completion_logprob) in enumerate(
        zip(
            tqdm.tqdm(rollouts, desc="training"),
            advantages,
            old_completion_logprobs,
            strict=True,
        )
    ):
        new_completion_logprob: Float[Tensor, ""] = completion_logprob(model, rollout)

        loss: Float[Tensor, ""] = gspo_loss(
            new_completion_logprob=new_completion_logprob,
            old_completion_logprob=old_completion_logprob,
            advantage=advantage,
            completion_length=len(rollout.completion_tokens),
            cfg=cfg,
        )

        loss.backward()

        if i % cfg.train_batch_size == 0 or i == len(rollouts) - 1:
            optimizer.step()
            optimizer.zero_grad()


def gspo_loss(
    new_completion_logprob: Float[Tensor, ""],
    old_completion_logprob: Float[Tensor, ""],
    advantage: float,
    completion_length: int,
    cfg: GSPOConfig,
) -> None:
    probability_ratio: Float[Tensor, ""] = torch.exp(
        (new_completion_logprob - old_completion_logprob) / completion_length
    )

    clipped_probability_ratio: Float[Tensor, ""] = torch.clamp(
        probability_ratio, min=1 - cfg.clip_epsilon_low, max=1 + cfg.clip_epsilon_high
    )

    return -torch.min(
        advantage * probability_ratio, advantage * clipped_probability_ratio
    )


def is_prefix(prefix: list[int], whole: list[int]) -> bool:
    if len(prefix) > len(whole):
        return False
    return all(x == y for x, y in zip(prefix, whole))


async def ground_truth_reward_function(completion: str, extra_data: Any) -> float:
    true_answer = extra_data["true_answer"]  # extra_data['true_answer']
    question = extra_data["question"]

    reward_model_prompt = DEFAULT_GT_TEMPLATE.format(
        question=question,
        answer=completion,
        criterion=true_answer,
        instructions=DEFAULT_GT_INSTRUCTIONS,
    )

    client = AsyncOpenAI()

    for i_retry in count():
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": reward_model_prompt}],
            )
        except Exception as e:
            delay = 2**i_retry
            print(
                f"OpenAI call failed on retry {i_retry}. Waiting for {delay} seconds and trying again. The exception is: {e}"
            )
            await asyncio.sleep(delay)
            continue
        break

    openai_output = response.choices[0].message.content

    # parse the response
    # extract the grade
    match = None
    for m in re.finditer(r"GRADE:(.*)", openai_output):
        match = m
    if match:
        value = match.group(1)
        value = "".join(
            filter(str.isdigit, value)
        )  # remove anything that's not a digit
    else:
        # if no grade is found, return 0 and log a warning
        # try look for 'Grade: '
        match = None
        for m in re.finditer(r"Grade:(.*)", openai_output):
            match = m
        if match:
            value = match.group(1)
            value = "".join(
                filter(str.isdigit, value)
            )  # remove anything that's not a digit
        else:
            value = ""

    if value is None or value == "":
        return 0.0
        # TODO: maybe this should be return None, but this requires adding support for missing rewards
    else:
        value = float(value)
        if value > 10.0:
            print(f"Value is greater than 10.0: {value}, clipping to 10.0")
            value = 10.0
        elif value < 0.0:
            print(f"Value is less than 0.0: {value}, clipping to 0.0")
            value = 0.0

    return float(value) / 10.0


async def dummy_reward_function(completion: str, extra_data: Any) -> float:
    return float(completion.count("."))


def main_dummy() -> None:
    dataset = [
        Datapoint(
            prompt=[{"role": "user", "content": "Please say something"}],
            extra_data=None,
        )
        for _ in range(4096)
    ]

    gspo_train(cfg=GSPOConfig(), dataset=dataset, reward_function=dummy_reward_function)


def main_old() -> None:
    olympiads_dataset = pd.read_csv("data/olympiads.csv")

    dataset: list[Datapoint] = [
        Datapoint(
            prompt=[
                {
                    "role": "user",
                    "content": AQUARAT_TEMPLATE_STYLIZED_RED_TEAM.format(
                        incorrect_answer=example["stored_incorrect_answer"]
                    ),
                }
            ],
            extra_data={
                "stored_incorrect_answer": example.get("stored_incorrect_answer", None),
                "true_answer": example.get("target", None),
                "question": example.get("question", None),
            },
        )
        for example in olympiads_dataset.to_dict("records")
    ]

    cfg = GSPOConfig()

    tokenizer = load_tokenizer(cfg)

    completions = generate_completions(
        [
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": f"What is 2 * {i}."},
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
            for i in range(2)
        ],
        None,
        cfg,
    )

    print(f"{completions=}")

    exit()

    download_gguf_model(cfg)

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
    main_dummy()


# ruff: noqa: F722
