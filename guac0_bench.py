import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt2")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--dtype", type=str, default="auto")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--use-fast", action="store_true")
parser.add_argument("--add-special-tokens", action="store_true")
args = parser.parse_args()


def main():
    # Call python main.py script
    subprocess.run([
        "python",
        "main.py",
        "--tasks", "sciq,lambada_openai,piqa,siqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,boolq",
        "--model", "gpt2",
        "--model_args", f"pretrained={args.model},dtype={args.dtype},trust_remote_code=True,low_cpu_mem_usage=True,use_fast={args.use_fast},add_special_tokens={args.add_special_tokens}",
        "--batch_size", f"{args.batch_size}",
        "--output_path", f"results/{args.model.replace('/', '_')}.json",
        "--device", args.device,
        "--num_fewshot", "0",
        "--no_cache",
    ])


if __name__ == "__main__":
    main()

