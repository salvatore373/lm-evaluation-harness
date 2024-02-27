"""
Usage:
python multilingual_benchmark_view.py --dir "results/"
"""

import argparse
import json
import os

import numpy as np
from pytablewriter import MarkdownTableWriter


# Creates markdown table for the given directory of lm-eval results


parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="directory to list", default="./results")
args = parser.parse_args()


def read_json(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data


def main(args):
    file_paths = [fp for fp in os.listdir(args.dir) if fp.endswith(".json")]
    print(file_paths)

    task2name = {
        "arc_challenge": "ARC Challenge✱",
        "arc_easy": "ARC Easy✱",
        "lambada_openai": "LAMBADA OpenAI",
        "hellaswag": "HellaSwag✱",
        "piqa": "PIQA",
        "sciq": "SciQ",
        "truthfulqa_mc": "TruthfulQA (MC)",
        "winogrande": "Winogrande",
        "arc_challenge_mt_de": "ARC Challenge✱ (DE)",
        "lambada_openai_mt_de": "LAMBADA OpenAI (DE)",
        "hellaswag_mt_de": "HellaSwag✱ (DE)",
        "truthfulqa_mc_mt_de": "TruthfulQA (MC) (DE)",
        "hendrycksTest_mt_de": "Hendrycks Test (DE)",
        "arc_challenge_mt_es": "ARC Challenge✱ (ES)",
        "lambada_openai_mt_es": "LAMBADA OpenAI (ES)",
        "hellaswag_mt_es": "HellaSwag✱ (ES)",
        "truthfulqa_mc_mt_es": "TruthfulQA (MC) (ES)",
        "hendrycksTest_mt_es": "Hendrycks Test (ES)",
        "arc_challenge_mt_fr": "ARC Challenge✱ (FR)",
        "lambada_openai_mt_fr": "LAMBADA OpenAI (FR)",
        "hellaswag_mt_fr": "HellaSwag✱ (FR)",
        "truthfulqa_mc_mt_fr": "TruthfulQA (MC) (FR)",
        "hendrycksTest_mt_fr": "Hendrycks Test (FR)",
        "arc_challenge_mt_it": "ARC Challenge✱ (IT)",
        "lambada_openai_mt_it": "LAMBADA OpenAI (IT)",
        "hellaswag_mt_it": "HellaSwag✱ (IT)",
        "truthfulqa_mc_mt_it": "TruthfulQA (MC) (IT)",
        "hendrycksTest_mt_it": "Hendrycks Test (IT)",
        "arc_challenge_mt_nl": "ARC Challenge✱ (NL)",
        "hellaswag_mt_nl": "HellaSwag✱ (NL)",
        "truthfulqa_mc_mt_nl": "TruthfulQA (MC) (NL)",
        "hendrycksTest_mt_nl": "Hendrycks Test (NL)",
        "arc_challenge_mt_pt": "ARC Challenge✱ (PT)",
        "hellaswag_mt_pt": "HellaSwag✱ (PT)",
        "truthfulqa_mc_mt_pt": "TruthfulQA (MC) (PT)",
        "hendrycksTest_mt_pt": "Hendrycks Test (PT)",
    }
    task2metric = {
        "arc_challenge": "acc_norm",
        "arc_easy": "acc_norm",
        "lambada_openai": "acc",
        "hellaswag": "acc_norm",
        "piqa": "acc",
        "sciq": "acc",
        "truthfulqa_mc": "mc2",
        "winogrande": "acc",
        "arc_challenge_mt_de": "acc_norm",
        "lambada_openai_mt_de": "acc",
        "hellaswag_mt_de": "acc_norm",
        "truthfulqa_mc_mt_de": "mc2",
        "hendrycksTest_mt_de": "acc",
        "arc_challenge_mt_es": "acc_norm",
        "lambada_openai_mt_es": "acc",
        "hellaswag_mt_es": "acc_norm",
        "truthfulqa_mc_mt_es": "mc2",
        "hendrycksTest_mt_es": "acc",
        "arc_challenge_mt_fr": "acc_norm",
        "lambada_openai_mt_fr": "acc",
        "hellaswag_mt_fr": "acc_norm",
        "truthfulqa_mc_mt_fr": "mc2",
        "hendrycksTest_mt_fr": "acc",
        "arc_challenge_mt_it": "acc_norm",
        "lambada_openai_mt_it": "acc",
        "hellaswag_mt_it": "acc_norm",
        "truthfulqa_mc_mt_it": "mc2",
        "hendrycksTest_mt_it": "acc",
        "arc_challenge_mt_nl": "acc_norm",
        "hellaswag_mt_nl": "acc_norm",
        "truthfulqa_mc_mt_nl": "mc2",
        "hendrycksTest_mt_nl": "acc",
        "arc_challenge_mt_pt": "acc_norm",
        "hellaswag_mt_pt": "acc_norm",
        "truthfulqa_mc_mt_pt": "mc2",
        "hendrycksTest_mt_pt": "acc",
    }
    task_headers = []
    for task in sorted(task2name.keys()):
        task_headers.append(f"{task2name[task]}")
    headers = ["Model", "Average", *task_headers]
    rows = []
    for result in file_paths:
        data = read_json(os.path.join(args.dir, result))
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
            score = f"{data['results'][task][task2metric[task]] * 100:.2f}%"
            row.append(score)

        rows.append(row)

    # Sort by average accuracy
    rows = sorted(rows, key=lambda x: float(x[1]), reverse=True)
    print(rows)

    # Print table in markdown
    writer = MarkdownTableWriter(
        table_name="0-shot Multilingual Results",
        headers=headers,
        value_matrix=rows,
        margin=1,
        flavor="github",
    )
    writer.dump("multilingual_benchmark_results.md")
    with open("multilingual_benchmark_results.md", "a") as f:
        f.write("\n\* : Byte-length Normalized Accuracy\n")


if __name__ == "__main__":
    main(args)
