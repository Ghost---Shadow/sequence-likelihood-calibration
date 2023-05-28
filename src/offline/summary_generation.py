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

from wrapped_datasets.sft_dataset import SftDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-checkpoint", type=str, default="t5-small")
    parser.add_argument("--split", type=str)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./generated_data/generated_summaries",
    )
    parser.add_argument("--num-return-sequences", type=int, default=5)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    return args


def generate_summaries(
    summary_checkpoint, output_dir, split, num_return_sequences, debug
):
    device = "cuda"

    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    summary_model = T5ForConditionalGeneration.from_pretrained(summary_checkpoint)
    summary_model.eval()

    dataset = SftDataset(split, debug=debug)
    data_loader = DataLoader(
        dataset,
        batch_size=1,  # Dont change
        shuffle=False,
        collate_fn=SftDataset.collate_fn,
    )

    summary_model = summary_model.to(device)

    with open(output_dir / f"{split}.jsonl", "w") as f:
        for batch in tqdm(data_loader):
            input_ids = batch["input_ids"].to(device)

            # Generate summaries
            with torch.no_grad():
                generated_ids = summary_model.generate(
                    input_ids,
                    max_length=512,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                )
                generated_summaries = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                input_prompt = tokenizer.batch_decode(
                    input_ids, skip_special_tokens=True
                )

            f.write(
                json.dumps(
                    {
                        "prompt": input_prompt[0],
                        "summaries": generated_summaries,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    args = parse_args()

    summary_checkpoint = args.summary_checkpoint
    output_dir = args.output_dir
    num_return_sequences = args.num_return_sequences
    split = args.split
    debug = args.debug

    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    generate_summaries(
        summary_checkpoint, output_dir, split, num_return_sequences, debug
    )
