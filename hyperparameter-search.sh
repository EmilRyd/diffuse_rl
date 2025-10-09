CUDA_VISIBLE_DEVICES=0 uv run gpt_oss_unsloth_rl.py --learning-rate 5e-5 &
CUDA_VISIBLE_DEVICES=1 uv run gpt_oss_unsloth_rl.py --learning-rate 2e-4 &
CUDA_VISIBLE_DEVICES=2 uv run gpt_oss_unsloth_rl.py --learning-rate 8e-4 &
CUDA_VISIBLE_DEVICES=3 uv run gpt_oss_unsloth_rl.py --learning-rate 1.6e-3 &
CUDA_VISIBLE_DEVICES=4 uv run gpt_oss_unsloth_rl.py --learning-rate 2.5e-5 &
CUDA_VISIBLE_DEVICES=5 uv run gpt_oss_unsloth_rl.py --learning-rate 1.25e-5 &
CUDA_VISIBLE_DEVICES=6 uv run gpt_oss_unsloth_rl.py --learning-rate 6.25e-6 &
CUDA_VISIBLE_DEVICES=7 uv run gpt_oss_unsloth_rl.py --learning-rate 3.125e-6 &
wait
