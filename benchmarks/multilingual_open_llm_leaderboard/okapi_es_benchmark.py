"""
Usage: Run from project root directory with the following command:

```sh
python benchmarks/multilingual_open_llm_leaderboard/okapi_es_benchmark.py \
    --model "stabilityai/stablelm-3b-4e1t" \
    --batch_size 8
```
"""
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt2")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--dtype", type=str, default="auto")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--use-fast", action="store_true")
args = parser.parse_args()


def main():
    # Call python main.py script
    subprocess.run(
        [
            "python",
            "main.py",
            "--tasks",
            "arc_challenge_mt_es,hellaswag_mt_es,hendrycksTest_mt_es,truthfulqa_mc_mt_es",
            "--model",
            "hf-causal-experimental",
            "--model_args",
            f"pretrained={args.model},dtype={args.dtype},trust_remote_code=True,low_cpu_mem_usage=True,use_fast={args.use_fast}",
            "--batch_size",
            f"{args.batch_size}",
            "--output_path",
            f"results/{args.model.replace('/', '_')}.json",
            "--device",
            args.device,
            "--num_fewshot",
            "0",
            "--no_cache",
        ]
    )


if __name__ == "__main__":
    main()