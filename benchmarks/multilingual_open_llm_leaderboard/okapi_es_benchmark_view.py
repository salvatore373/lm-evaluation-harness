"""
Usage: Run from project root directory with the following command:

```sh
python benchmarks/multilingual_open_llm_leaderboard/okapi_es_benchmark_view.py \
    --results_dir ./results \
    --outputs_dir benchmarks/multilingual_open_llm_leaderboard
```
"""
import argparse
import json
import os

import numpy as np
from pytablewriter import MarkdownTableWriter


# Creates markdown table for the given directory of lm-eval results


parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", help="directory to list", required=True)
parser.add_argument("--outputs_dir", help="directory to list", required=True)
args = parser.parse_args()


def read_json(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data


def main(args):
    file_paths = [fp for fp in os.listdir(args.results_dir) if fp.endswith(".json")]
    print(file_paths)

    task2name = {
        "arc_challenge_mt_es": "ARC Challenge (es)",
        "hellaswag_mt_es": "HellaSwag (es)",
        "hendrycksTest_mt_es": "MMLU (es)",
        "truthfulqa_mc_mt_es": "TruthfulQA (es)",
    }
    task2metric = {
        "arc_challenge_mt_es": "acc_norm",
        "hellaswag_mt_es": "acc_norm",
        "hendrycksTest_mt_es": "acc",
        "truthfulqa_mc_mt_es": "mc2",
    }
    task_headers = []
    for task in sorted(task2name.keys()):
        task_headers.append(f"{task2name[task]}<br>({task2metric[task]})")
        # task_headers.append(f"{task2name[task]}")
    headers = ["Model", "Average", *task_headers]
    rows = []
    for result in file_paths:
        data = read_json(os.path.join(args.results_dir, result))
        model_name = [
            k for k in data["config"]["model_args"].split(",") if "pretrained" in k
        ][0].split("=")[1]
        row = [model_name]

        # Compute mean accuracy
        sum = 0
        for task in sorted(task2name.keys()):
            sum += data["results"][task][task2metric[task]]
        row.append(f"{(sum / len(task2name)) * 100.0:.2f}")

        for task in sorted(task2name.keys()):
            # score = f"{data['results'][task]['acc'] * 100:.2f} ± {data['results'][task]['acc_stderr'] * 100:.2f}"
            score = f"{data['results'][task][task2metric[task]] * 100:.2f}"
            row.append(score)

        rows.append(row)

    # Sort by average accuracy
    rows = sorted(rows, key=lambda x: float(x[1]), reverse=True)
    print(rows)

    # Print table in markdown
    writer = MarkdownTableWriter(
        table_name="Results",
        headers=headers,
        value_matrix=rows,
        margin=1,
        flavor="github",
    )
    writer.dump(os.path.join(args.outputs_dir, "okapi_es_benchmark_results.md"))


if __name__ == "__main__":
    main(args)
