import json
from pathlib import Path
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer
from datasets import load_dataset
from wrapped_datasets.utils import clean_prompt, test_dataloader


class SlicDataset(Dataset):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    INSTRUCTION = "summarize: "

    def __init__(self, split, jsonl_path, debug=False):
        SlicDataset.tokenizer.add_tokens(["\n"])

        # Store references from HF dataset
        self.hf_dataset = load_dataset("CarperAI/openai_summarize_tldr")[split]
        prompt_lut = {}
        for row in tqdm(self.hf_dataset):
            prompt = clean_prompt(row["prompt"])
            prompt_lut[prompt] = {
                "reference": row["label"],
            }

        # Load generated summaries
        self.dataset = []
        with open(jsonl_path) as f:
            for line in f:
                self.dataset.append(json.loads(line))

        train_split = int(0.8 * len(self.dataset))

        if split == "train":
            self.dataset = self.dataset[:train_split]
        elif split == "valid":
            self.dataset = self.dataset[train_split:]
        else:
            raise NotImplementedError()

        if debug:
            self.dataset = self.dataset[:100]

        # Merge two datasets
        foo = []
        for i in tqdm(range(len(self.dataset))):
            prompt = self.dataset[i]["prompt"]
            prompt = clean_prompt(prompt)
            if prompt not in prompt_lut:
                continue
            hf_data = prompt_lut[prompt]
            foo.append({
                **self.dataset[i],
                **hf_data,
            })

        self.dataset = foo

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def filter_function(row):
        row = SlicDataset.format_row(row)
        all_text = row["full_prompt"] + row["correct_answer"]
        tokens = SlicDataset.tokenizer(
            all_text,
            return_tensors="pt",
        ).input_ids
        return len(tokens) <= 512

    def __getitem__(self, idx):
        row = self.dataset[idx]
        return SlicDataset.format_row(row)

    def sanity_check(self):
        prompts = [self[0]["full_prompt"]]
        input_ids = SlicDataset._tokenize(prompts)

        return (
            input_ids,
            self[0]["full_prompt"],
            self[0]["correct_answer"],
        )

    @staticmethod
    def _tokenize(s, padding=True):
        tokens = SlicDataset.tokenizer(
            s,
            padding=padding,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        return tokens

    @staticmethod
    def collate_fn(batch):
        input_ids = [item["full_prompt"] for item in batch]
        labels = [item["correct_answer"] for item in batch]

        input_ids = SlicDataset._tokenize(input_ids)
        labels = SlicDataset._tokenize(labels, padding=False)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


if __name__ == "__main__":
    jsonl_path = "generated_data/classified_summaries_length/result.jsonl"
    dataset = SlicDataset(jsonl_path=jsonl_path, split="valid", debug=True)
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=SlicDataset.collate_fn
    )

    sanity_path = Path("generated_data/sanity_check")
    sanity_path.mkdir(exist_ok=True, parents=True)

    tokenizer = SlicDataset.tokenizer

    test_dataloader(dataloader, sanity_path / "slic.txt", tokenizer)
