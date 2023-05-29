import argparse
import os
from pathlib import Path
import json

import tqdm


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
        default="./generated_data/classified_summaries",
    )

    args = parser.parse_args()
    return args


def better_than(model, left, right):
    ...


def worse_than(model, left, right):
    return not better_than(model, left, right)


def find_best_and_worst(model, strings):
    best = worst = strings[0]

    for i in tqdm(range(0, len(strings), 2), leave=False):
        # If there's only one string left, compare it with the current best and worst
        if i == len(strings) - 1:
            if better_than(model, strings[i], best):
                best = strings[i]
            elif worse_than(model, strings[i], worst):
                worst = strings[i]
            break

        # Compare the two strings and update the best and worst
        if better_than(model, strings[i], strings[i + 1]):
            if better_than(model, strings[i], best):
                best = strings[i]
            elif worse_than(model, strings[i + 1], worst):
                worst = strings[i + 1]
        else:
            if better_than(model, strings[i + 1], best):
                best = strings[i + 1]
            elif worse_than(model, strings[i], worst):
                worst = strings[i]

    return best, worst


def classify_summaries(summary_dir, classification_dir):
    for file_name in os.listdir(summary_dir):
        with open(classification_dir / file_name, "w") as f_out:
            with open(summary_dir / file_name) as f_in:
                for line in f_in:
                    row = json.loads(line)
                    best, worst = find_best_and_worst(row["summaries"])
                    s = json.dumps(
                        {
                            "prompt": row["prompt"],
                            "chosen": best,
                            "rejected": worst,
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
