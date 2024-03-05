"""
Usage: Run from project root directory with the following command:

```sh
python multilingual_benchmark.py --model "stabilityai/stablelm-2-1_6b"
```
"""
import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt2")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--dtype", type=str, default="auto")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--use-fast", action="store_true")
parser.add_argument("--add-special-tokens", action="store_true")
args = parser.parse_args()


def main():
    results_file = os.path.join("results", f"{args.model.replace('/', '_')}.json")
    results_file = os.path.abspath(results_file)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    assert not os.path.exists(results_file), f"{results_file} results already exists! Check the `results` dir!"
    subprocess.run(
        [
            "python",
            "main.py",
            "--tasks",
            "hendrycksTest-*,lambada_openai,arc_easy,arc_challenge,hellaswag,piqa,sciq,winogrande,truthfulqa_mc,arc_challenge_mt_de,lambada_openai_mt_de,hellaswag_mt_de,truthfulqa_mc_mt_de,hendrycksTest_mt_de,arc_challenge_mt_es,lambada_openai_mt_es,hellaswag_mt_es,truthfulqa_mc_mt_es,hendrycksTest_mt_es,arc_challenge_mt_fr,lambada_openai_mt_fr,hellaswag_mt_fr,truthfulqa_mc_mt_fr,hendrycksTest_mt_fr,arc_challenge_mt_it,lambada_openai_mt_it,hellaswag_mt_it,truthfulqa_mc_mt_it,hendrycksTest_mt_it,arc_challenge_mt_pt,hellaswag_mt_pt,truthfulqa_mc_mt_pt,hendrycksTest_mt_pt,arc_challenge_mt_nl,hellaswag_mt_nl,truthfulqa_mc_mt_nl,hendrycksTest_mt_nl",
            "--model",
            "hf-causal-experimental",
            "--model_args",
            f"pretrained={args.model},trust_remote_code=True,add_special_tokens={args.add_special_tokens},dtype={args.dtype}",
            "--batch_size",
            f"{args.batch_size}",
            "--output_path",
            results_file,
            "--device",
            args.device,
            "--num_fewshot",
            "0",
            "--no_cache",
        ]
    )


if __name__ == "__main__":
    main()
