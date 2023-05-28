from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
from datasets import load_dataset
from wrapped_datasets.utils import clean_prompt, test_dataloader


class SftDataset(Dataset):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    INSTRUCTION = "summarize: "

    def __init__(self, split, debug=False):
        SftDataset.tokenizer.add_tokens(["\n"])
        self.dataset = load_dataset("CarperAI/openai_summarize_tldr")[split]
        if debug:
            self.dataset = self.dataset.select(range(5))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        prompt = self.dataset[idx]["prompt"]
        prompt = SftDataset.INSTRUCTION + clean_prompt(prompt)
        return {
            "full_prompt": prompt + "\n",
            "label": self.dataset[idx]["label"],
        }

    def sanity_check(self):
        prompts = [self[0]["prompt"]]
        input_ids = SftDataset._tokenize(prompts)

        return input_ids, prompts[0], self[0]["label"]

    @staticmethod
    def _tokenize(s):
        tokens = SftDataset.tokenizer(
            s,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        return tokens

    @staticmethod
    def collate_fn(batch):
        prompts = [item["full_prompt"] for item in batch]
        labels = [item["label"] for item in batch]

        input_ids = SftDataset._tokenize(prompts)
        labels = SftDataset._tokenize(labels)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


if __name__ == "__main__":
    dataset = SftDataset(split="train")
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=SftDataset.collate_fn
    )

    sanity_path = Path("generated_data/sanity_check")
    sanity_path.mkdir(exist_ok=True, parents=True)

    tokenizer = SftDataset.tokenizer

    test_dataloader(dataloader, sanity_path / "sft.txt", tokenizer)
