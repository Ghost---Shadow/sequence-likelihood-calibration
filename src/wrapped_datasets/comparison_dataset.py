import json
from pathlib import Path
import random
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
from datasets import load_dataset
from wrapped_datasets.utils import clean_prompt, test_dataloader


class ComparisionDataset(Dataset):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    INSTRUCTION = "summarize: "
    LABELS = ["A", "B"]

    def _jsonl_loader(self, jsonl_path, split, debug):
        self.dataset = []
        with open(jsonl_path) as f:
            for line in f:
                self.dataset.append(json.loads(line))

        train_split = int(0.8 * len(self.dataset))

        if split == "train":
            self.dataset = self.dataset[:train_split]
        elif split == "valid1":
            self.dataset = self.dataset[train_split:]
        else:
            raise NotImplementedError()

        if debug:
            self.dataset = self.dataset[:100]

    def _huggingface_loader(self, split, debug):
        self.dataset = load_dataset("CarperAI/openai_summarize_comparisons")[split]
        # self.dataset = self.dataset.filter(ComparisionDataset.filter_function)
        if debug:
            self.dataset = self.dataset.select(range(100))

    def __init__(self, split, jsonl_path=None, debug=False):
        ComparisionDataset.tokenizer.add_tokens(["\n"])

        if jsonl_path is None:
            self._huggingface_loader(split, debug)
        else:
            self._jsonl_loader(jsonl_path, split, debug)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def filter_function(row):
        row = ComparisionDataset.format_row(row)
        all_text = row["full_prompt"] + row["correct_answer"]
        tokens = ComparisionDataset.tokenizer(
            all_text,
            return_tensors="pt",
        ).input_ids
        return len(tokens) <= 512

    @staticmethod
    def tokenized_labels():
        tokens = ComparisionDataset.tokenizer(
            ComparisionDataset.LABELS,
            padding=False,
            truncation=False,  # Manual truncation
            return_tensors="pt",
        ).input_ids
        # Drop EOS
        return tokens[:, 0].squeeze()

    @staticmethod
    def format_row(row, correct_index=None):
        row["chosen"] = row["chosen"].replace("TL;DR: ", "").strip()
        row["rejected"] = row["rejected"].replace("TL;DR: ", "").strip()
        answers = [None, None]
        prompt = row["prompt"]

        prompt = clean_prompt(prompt)

        if correct_index is None:
            correct_index = random.randint(0, 1)

        correct_answer = ComparisionDataset.LABELS[correct_index]
        answers[correct_index] = row["chosen"]
        answers[1 - correct_index] = row["rejected"]

        # Create the full prompt
        full_prompt = f"{prompt}\nA: {answers[0]}\nB: {answers[1]}"
        full_prompt = full_prompt.strip()
        full_prompt = f"{full_prompt}\n"

        return {
            "full_prompt": full_prompt,
            "correct_answer": correct_answer,
        }

    def __getitem__(self, idx):
        row = self.dataset[idx]
        return ComparisionDataset.format_row(row)

    def sanity_check(self):
        prompts = [self[0]["full_prompt"]]
        input_ids = ComparisionDataset._tokenize(prompts)

        return (
            input_ids,
            self[0]["full_prompt"],
            self[0]["correct_answer"],
        )

    @staticmethod
    def _tokenize(s, padding=True):
        tokens = ComparisionDataset.tokenizer(
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

        input_ids = ComparisionDataset._tokenize(input_ids)
        labels = ComparisionDataset._tokenize(labels, padding=False)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


if __name__ == "__main__":
    dataset = ComparisionDataset(split="train", debug=True)
    print("Label tokens", dataset.tokenized_labels())
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=ComparisionDataset.collate_fn
    )

    sanity_path = Path("generated_data/sanity_check")
    sanity_path.mkdir(exist_ok=True, parents=True)

    tokenizer = ComparisionDataset.tokenizer

    test_dataloader(dataloader, sanity_path / "hf_comparison.txt", tokenizer)

    jsonl_path = "generated_data/classified_summaries_length/result.jsonl"

    dataset = ComparisionDataset(jsonl_path=jsonl_path, split="train", debug=True)
    print("Label tokens", dataset.tokenized_labels())
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=ComparisionDataset.collate_fn
    )
    test_dataloader(dataloader, sanity_path / "length_train.txt", tokenizer)

    dataset = ComparisionDataset(jsonl_path=jsonl_path, split="valid1", debug=True)
    print("Label tokens", dataset.tokenized_labels())
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=ComparisionDataset.collate_fn
    )
    test_dataloader(dataloader, sanity_path / "length_valid1.txt", tokenizer)
