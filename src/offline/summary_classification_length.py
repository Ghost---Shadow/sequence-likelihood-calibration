import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from torch.utils.data import DataLoader
import torch
import json
from wrapped_datasets.comparison_dataset import ComparisionDataset

from wrapped_datasets.sft_dataset import SftDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-dir",
        type=str,
        default="./generated_data/generated_summaries",
    )
    parser.add_argument(
        "--classification-dir",
        type=str,
        default="./generated_data/classified_summaries_length",
    )

    args = parser.parse_args()
    return args


def find_longest_and_shortest(strings):
    if not strings:
        return None, None

    longest = shortest = strings[0]

    for i in range(0, len(strings), 2):
        # If there's only one string left, compare it with the current longest and shortest
        if i == len(strings) - 1:
            if len(strings[i]) > len(longest):
                longest = strings[i]
            elif len(strings[i]) < len(shortest):
                shortest = strings[i]
            break

        # Compare the two strings and update the longest and shortest
        if len(strings[i]) > len(strings[i + 1]):
            if len(strings[i]) > len(longest):
                longest = strings[i]
            if len(strings[i + 1]) < len(shortest):
                shortest = strings[i + 1]
        else:
            if len(strings[i + 1]) > len(longest):
                longest = strings[i + 1]
            if len(strings[i]) < len(shortest):
                shortest = strings[i]

    return longest, shortest


def classify_summaries(summary_dir, classification_dir):
    with open(classification_dir / "result.jsonl", "w") as f_out:
        with open(summary_dir / "result.jsonl") as f_in:
            for line in f_in:
                row = json.loads(line)
                longest, shortest = find_longest_and_shortest(row["summaries"])
                s = json.dumps(
                    {
                        "prompt": row["prompt"],
                        "chosen": shortest,
                        "rejected": longest,
                    }
                )
                f_out.write(f"{s}\n")
                f_out.flush()


if __name__ == "__main__":
    args = parse_args()

    summary_dir = args.summary_dir
    classification_dir = args.classification_dir

    summary_dir = Path(summary_dir)
    classification_dir = Path(classification_dir)

    classification_dir.mkdir(parents=True, exist_ok=True)

    classify_summaries(summary_dir, classification_dir)
