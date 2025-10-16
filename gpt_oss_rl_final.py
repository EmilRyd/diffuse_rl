from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import get_peft_model, LoraConfig, PeftModel
import torch
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import Optimizer, AdamW
from dotenv import load_dotenv
from time import perf_counter
from subprocess import run, Popen, DEVNULL
import wget
import pandas as pd
import wandb
from statistics import mean, stdev
import aiohttp
import gc
from more_itertools import chunked
import os
from os import makedirs
from itertools import count, chain
import random
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
import enum

load_dotenv()
"""
from templates import (
    AQUARAT_TEMPLATE_STYLIZED_RED_TEAM,
    DEFAULT_GT_INSTRUCTIONS,
    DEFAULT_GT_TEMPLATE,
)
"""
from templates import (
    AQUARAT_TEMPLATE_STYLIZED_RED_TEAM,
    DEFAULT_GT_INSTRUCTIONS,
    DEFAULT_GT_TEMPLATE,
    BT_NORMAL_TM_TEMPLATE,
    GAMEABLE_TM_INSTRUCTIONS,
)

# reward function choices
class RewardType(enum.Enum):
    GROUND_TRUTH = "ground_truth" # sparse
    TRUSTED_MONITOR = "trusted_monitor" # dense

def fetch_submission(full_submission: str) -> str:

    if "final<|message|>" in full_submission:
        submission = full_submission.split("final<|message|>")[-1].strip()
        return submission
    if "</think>" in full_submission:
        submission = full_submission.split("</think>")[-1].strip()
        return submission
    else:
        return ''

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
    groups_per_epoch: int = 32
    group_size: int = 8
    train_batch_size: int = 64
    clip_epsilon_low: float = 3e-4
    clip_epsilon_high: float = 4e-4
    epochs: int = 1024
    gspo: bool = True
    reasoning_effort: str = "low"
    use_wandb: bool = True
    wandb_project: str | None = "gpt-oss-llama-cpp-rl"
    wandb_run_name: str | None = None

    def __post_init__(self) -> None:
        assert (self.groups_per_epoch * self.group_size) % self.train_batch_size == 0


@dataclass(frozen=True, slots=True)
class Datapoint:
    prompt: list[dict[str, str]]
    extra_data: Any = None


def load_huggingface_transformer(rank: int, cfg: GSPOConfig) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(
        cfg.huggingface_model_name,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    ).cuda(rank)

    model.train()

    peft_config = peft_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=2 * cfg.lora_rank,
        target_modules="all-linear",
    )

    model = get_peft_model(model, peft_config)

    return DistributedDataParallel(
        model,
        device_ids=[rank],
        find_unused_parameters=False,  # TODO: check if i need find_unused_parameters=True
    )


def load_tokenizer(cfg: GSPOConfig) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(cfg.huggingface_model_name)


def download_gguf_model(cfg: GSPOConfig) -> None:
    if isfile(cfg.gguf_model_path):
        return
    makedirs("weights", exist_ok=True)
    wget.download(cfg.gguf_model_url, cfg.gguf_model_path)


def make_optimizer(model: PeftModel, cfg: GSPOConfig) -> Optimizer:
    trainable_params = [
        param for param in model.module.parameters() if param.requires_grad
    ]
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
    rank: int,
    gguf_lora_adapter_filename: str | None,
    n_parallel_sequences: int,
    per_sequence_context_length: int,
    port: int,
    cfg: GSPOConfig,
) -> Popen:
    gc.collect()
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated(rank)
    print(f"Allocated memory on cuda:{rank}: {allocated / 1024**3:.2f} Gib")

    process = Popen(
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
        ]
        + (
            ["--lora", gguf_lora_adapter_filename]
            if gguf_lora_adapter_filename is not None
            else []
        ),
        # stdout=DEVNULL,
        # stderr=DEVNULL,
        env={"CUDA_VISIBLE_DEVICES": str(rank)} | os.environ,
    )

    import time

    time.sleep(16)

    return process


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
                    "temperature": 1.0,
                },
            ) as resp:
                result = await resp.json()

        if "error" in result.keys():
            print(f"{result['error']=}")
            await asyncio.sleep(10 + random.random())
            continue

        """
        server_still_loading: bool = (
            "error" in result.keys() and result["error"]["message"] == "Loading model"
        )
        if server_still_loading:
            await asyncio.sleep(1)
            continue
        """

        text: str = result["choices"][0]["text"]
        print(text)
        return text


def generate_completions(
    rank: int,
    prompts_with_chat_template: list[list[int]],
    gguf_lora_adapter_filename: str | None,
    cfg: GSPOConfig,
) -> list[str]:
    server_port: int = cfg.llama_cpp_server_first_port + rank

    server_process: Popen = start_llama_cpp_server(
        rank=rank,
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
    rank: int,
    gguf_lora_adapter_filename: str | None,
    tokenizer: PreTrainedTokenizer,
    data: list[Datapoint],
    reward_function: Callable[[str, Any], Awaitable[float]],
    cfg: GSPOConfig,
) -> list[Rollout]:
    prompts: list[str] = [
        tokenizer.apply_chat_template(
            datapoint.prompt,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=cfg.reasoning_effort,
        )
        for datapoint in data
    ]

    completions: list[str] = generate_completions(
        rank=rank,
        prompts_with_chat_template=prompts,
        gguf_lora_adapter_filename=gguf_lora_adapter_filename,
        cfg=cfg,
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

    assert all(isinstance(reward, float) for reward in rewards)

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


def setup_distributed_data_parallel(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"  # wtf is this?
    os.environ["MASTER_PORT"] = "12355"  # wtf is this?
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device("cuda", rank),
    )
    dist.barrier()  # do i need barrier here?


def concatenate_from_all_processes(xs: list[Any], world_size: int) -> list[Any]:
    xs_from_all_processes: list[list | None] = [None] * world_size
    dist.all_gather_object(xs_from_all_processes, xs)
    assert all(xs is not None for xs in xs_from_all_processes)
    return list(chain.from_iterable(xs_from_all_processes))  # type: ignore


def gspo_train_process(
    rank: int,
    world_size: int,
    cfg: GSPOConfig,
    dataset: list[Datapoint],
    reward_function: Callable[[str, Any], Awaitable[float]],
) -> None:
    setup_distributed_data_parallel(rank=rank, world_size=world_size)

    assert cfg.groups_per_epoch % world_size == 0
    assert cfg.train_batch_size % world_size == 0

    main_process: bool = rank == 0

    if main_process:
        download_gguf_model(cfg)

    model: PeftModel = load_huggingface_transformer(rank=rank, cfg=cfg)
    tokenizer: PreTrainedTokenizer = load_tokenizer(cfg)
    optimizer: Optimizer = make_optimizer(model, cfg)

    Random(42).shuffle(dataset)

    if main_process and cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name)

    for epoch in tqdm.tqdm(range(cfg.epochs), desc="gspo training"):
        batch_data: list[Datapoint] = [
            dataset[i % len(dataset)]
            for i in range(
                cfg.groups_per_epoch * epoch, cfg.groups_per_epoch * (epoch + 1)
            )
            if i % world_size == rank
        ]

        grouped_data: list[list[Datapoint]] = [
            [datapoint] * cfg.group_size for datapoint in batch_data
        ]

        rollouts: list[Rollout] = generate_rollouts(
            rank=rank,
            gguf_lora_adapter_filename=gguf_lora_adapter_filename(cfg, epoch=epoch - 1)
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

        loss_metrics: list[LossMetrics] = train(
            rank=rank,
            world_size=world_size,
            model=model,
            optimizer=optimizer,
            rollouts=rollouts,
            advantages=advantages,
            cfg=cfg,
        )

        dist.barrier()

        if main_process:
            save_huggingface_lora_adapter(
                model, huggingface_lora_adapter_path(epoch=epoch, cfg=cfg)
            )
            convert_lora_to_gguf(
                huggingface_adapter_path=huggingface_lora_adapter_path(
                    epoch=epoch, cfg=cfg
                ),
                gguf_adapter_filename=gguf_lora_adapter_filename(cfg=cfg, epoch=epoch),
                cfg=cfg,
            )

        dist.barrier()

        all_loss_metrics = concatenate_from_all_processes(
            loss_metrics, world_size=world_size
        )
        all_rollouts = concatenate_from_all_processes(rollouts, world_size=world_size)
        all_grouped_rewards = concatenate_from_all_processes(
            grouped_rewards, world_size=world_size
        )
        if main_process and cfg.use_wandb:
            plot_metrics_on_wandb(
                loss_metrics=all_loss_metrics,
                rollouts=all_rollouts,
                grouped_rewards=all_grouped_rewards,
            )


def plot_metrics_on_wandb(
    loss_metrics: list["LossMetrics"],
    rollouts: list[Rollout],
    grouped_rewards: list[list[float]],
) -> None:
    wandb.log(
        {
            "loss/loss": mean(metric.loss for metric in loss_metrics),
            "loss/fraction_clipped": mean(metric.fraction_clipped for metric in loss_metrics),
            "loss/mean_log_probability_ratio": mean(
                metric.mean_log_probability_ratio for metric in loss_metrics
            ),
            "loss/max_abs_log_probability_ratio": max(
                abs(metric.max_abs_log_probability_ratio) for metric in loss_metrics
            ),
            "reward/mean": mean(chain.from_iterable(grouped_rewards)),
            "group/std": mean(stdev(group) for group in grouped_rewards),
            "group/fraction_mixed": len(
                [group for group in grouped_rewards if not all_close(group)]
            )
            / len(grouped_rewards),
            "length/mean_generated": mean(
                len(rollout.completion_tokens) for rollout in rollouts
            ),
            "length/max_generated": max(
                len(rollout.completion_tokens) for rollout in rollouts
            ),
            "length/mean_prompt": mean(
                len(rollout.prompt_tokens) for rollout in rollouts
            ),
            "length/max_prompt": max(
                len(rollout.prompt_tokens) for rollout in rollouts
            ),
        }
    )


def all_close(xs: list[float], epsilon: float = 1e-5) -> bool:
    return max(xs) - min(xs) <= epsilon


def gspo_train(
    cfg: GSPOConfig,
    dataset: list[Datapoint],
    reward_function: Callable[[str, Any], Awaitable[float]],
    world_size: int | None = None,
) -> None:
    if world_size is None:
        world_size = torch.cuda.device_count()

    mp.spawn(
        gspo_train_process,
        args=(world_size, cfg, dataset, reward_function),
        nprocs=world_size,
    )


def huggingface_lora_adapter_path(epoch: int, cfg: GSPOConfig) -> str:
    return join(cfg.save_adapters_path, f"epoch-{epoch}")


def gguf_lora_adapter_filename(cfg: GSPOConfig, epoch: int) -> str:
    return join(cfg.save_adapters_path, f"adapter-{epoch}.gguf")


def save_huggingface_lora_adapter(model: PeftModel, path: str) -> None:
    model.module.save_pretrained(path)


def completion_logprobs(
    rank: int, model: PeftModel, rollout: Rollout
) -> Float[Tensor, " position"]:
    tokens: Int[Tensor, " position"] = torch.tensor(
        rollout.prompt_tokens + rollout.completion_tokens
    ).cuda(rank)
    in_tokens: Int[Tensor, " position"] = tokens[:-1]
    out_tokens: Int[Tensor, " position"] = tokens[1:]

    all_logits: Float[Tensor, "position vocabulary_size"] = model(
        in_tokens.unsqueeze(0)
    ).logits.squeeze(0)

    all_logprobs = all_logits.log_softmax(-1)

    logprobs: Float[Tensor, " position"] = all_logprobs[
        torch.arange(out_tokens.numel()).cuda(rank), out_tokens
    ]

    return logprobs[-len(rollout.completion_tokens) :]


def gspo_advantages(grouped_rewards: list[list[float]]) -> list[list[float]]:
    grouped_advantages: list[list[float]] = []
    for group in grouped_rewards:
        assert len(group) >= 2
        advantages: list[float] = []
        for reward in group:
            # advantages.append(reward - (sum(group) - reward) / (len(group) - 1))
            advantages.append((reward - mean(group)) / (stdev(group) + 1e-5))
        grouped_advantages.append(advantages)
    return grouped_advantages


@dataclass(frozen=True, slots=True)
class LossMetrics:
    loss: float
    fraction_clipped: float
    mean_log_probability_ratio: float
    max_abs_log_probability_ratio: float


def train(
    rank: int,
    world_size: int,
    model: PeftModel,
    optimizer: Optimizer,
    rollouts: list[Rollout],
    advantages: list[float],
    cfg: GSPOConfig,
) -> list[LossMetrics]:
    with torch.no_grad():
        all_old_completion_logprobs: list[Float[Tensor, " position"]] = [
            completion_logprobs(rank=rank, model=model, rollout=rollout)
            for rollout in tqdm.tqdm(rollouts, desc="computing old logprobs")
        ]

    all_metrics: list[LossMetrics] = []

    for i, (rollout, advantage, old_completion_logprobs) in enumerate(
        zip(
            tqdm.tqdm(rollouts, desc="training"),
            advantages,
            all_old_completion_logprobs,
            strict=True,
        )
    ):
        new_completion_logprobs: Float[Tensor, " position"] = completion_logprobs(
            rank=rank, model=model, rollout=rollout
        )

        loss: Float[Tensor, ""]
        loss, metrics = gspo_loss(
            new_completion_logprobs=new_completion_logprobs,
            old_completion_logprobs=old_completion_logprobs,
            advantage=advantage,
            cfg=cfg,
        )

        all_metrics.append(metrics)

        loss.backward()

        assert cfg.train_batch_size % world_size == 0
        if (i + 1) % (cfg.train_batch_size // world_size) == 0 or i == len(
            rollouts
        ) - 1:
            dist.barrier()
            torch.nn.utils.clip_grad_norm_(
                [param for param in model.module.parameters() if param.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            optimizer.zero_grad()

    return all_metrics


def gspo_loss(
    new_completion_logprobs: Float[Tensor, " position"],
    old_completion_logprobs: Float[Tensor, " position"],
    advantage: float,
    cfg: GSPOConfig,
) -> tuple[Float[Tensor, ""], LossMetrics]:
    log_probability_ratios: Float[Tensor, " #position"] = (
        new_completion_logprobs - old_completion_logprobs
    )

    if cfg.gspo:
        log_probability_ratios = log_probability_ratios.mean()

    probability_ratios: Float[Tensor, " #position"] = log_probability_ratios.exp()

    clipped_probability_ratios: Float[Tensor, " #position"] = torch.clamp(
        probability_ratios, min=1 - cfg.clip_epsilon_low, max=1 + cfg.clip_epsilon_high
    )

    loss = -torch.min(
        advantage * probability_ratios, advantage * clipped_probability_ratios
    ).mean()

    metrics = LossMetrics(
        loss=loss.item(),
        fraction_clipped=(
            advantage * probability_ratios > advantage * clipped_probability_ratios
        ).float().mean().item(),
        mean_log_probability_ratio=log_probability_ratios.mean().item(),
        max_abs_log_probability_ratio=log_probability_ratios.abs().max().item()
    )

    return loss, metrics


def is_prefix(prefix: list[int], whole: list[int]) -> bool:
    if len(prefix) > len(whole):
        return False
    return all(x == y for x, y in zip(prefix, whole))


"""
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
"""


async def ground_truth_reward_function(completion: str, extra_data: Any) -> float:
    true_answer = extra_data["true_answer"]  # extra_data['true_answer']
    question = extra_data["question"]

    answer_without_reasoning = fetch_submission(completion)

    # slightly ugly way of dealing with the fact that the templates have different inputs
    if REWARD_TYPE == RewardType.GROUND_TRUTH:
        reward_model_prompt = MONITOR_TEMPLATE.format(
            question=question,
            answer=answer_without_reasoning,
            criterion=true_answer,
            instructions=MONITOR_INSTRUCTIONS,
        )
    elif REWARD_TYPE == RewardType.TRUSTED_MONITOR:
        reward_model_prompt = MONITOR_TEMPLATE.format(
            question=question,
            solution=answer_without_reasoning,
            instructions=MONITOR_INSTRUCTIONS,
        )
    else:
        raise ValueError(f"Invalid monitor template: {MONITOR_TEMPLATE}")
    

    client = AsyncOpenAI()

    for i_retry in count():
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": reward_model_prompt}]
            )
        except Excpetion as e:
            delay = 2 ** i_retry
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
    else:
        value = float(value)
        if value > 10.0:
            print(f"Value is greater than 10.0: {value}, clipping to 10.0")
            value = 10.0
        elif value < 0.0:
            print(f"Value is less than 0.0: {value}, clipping to 0.0")
            value = 0.0

    return float(value) / 10.0


def load_olympiad_dataset() -> list[Datapoint]:
    olympiads_dataset = pd.read_csv("data/olympiads.csv")

    return [
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


def main() -> None:
    gspo_train(
        cfg=GSPOConfig(),
        dataset=load_olympiad_dataset(),
        # reward_function=dummy_reward_function,
        reward_function=ground_truth_reward_function,
    )


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


if __name__ == "__main__":

    # set the reward type. Edit this to change the reward type. if you want, you cna set this as an argument passed into the script
    REWARD_TYPE = RewardType.GROUND_TRUTH


    if REWARD_TYPE == RewardType.GROUND_TRUTH:
        MONITOR_TEMPLATE = DEFAULT_GT_TEMPLATE  # sparse (ground truth)
        MONITOR_INSTRUCTIONS = DEFAULT_GT_INSTRUCTIONS
    elif REWARD_TYPE == RewardType.TRUSTED_MONITOR:
        MONITOR_TEMPLATE = BT_NORMAL_TM_TEMPLATE # dense (proxy for ground truth)
        MONITOR_INSTRUCTIONS = GAMEABLE_TM_INSTRUCTIONS
    else:
        raise ValueError(f"Invalid reward type: {REWARD_TYPE}")

    main()
    # main_dummy()


# ruff: noqa: F722
