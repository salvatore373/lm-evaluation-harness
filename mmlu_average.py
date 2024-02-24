"""
Prints MMMLU (Hendrycks Test) average accuracy from a given results file
"""
import argparse
import json
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("file", help="directory to list", default="./results")
args = parser.parse_args()


def read_json(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data


def main(args):
    file_json = json.load(open(args.file))
    results = file_json["results"]
    all_acc = []
    for key, item in results.items():
        print(item)
        if key.startswith("hendrycksTest-"):
            all_acc.append(item["acc"])
    return np.mean(all_acc)


if __name__ == "__main__":
    avg_acc = main(args)
    print(avg_acc)
