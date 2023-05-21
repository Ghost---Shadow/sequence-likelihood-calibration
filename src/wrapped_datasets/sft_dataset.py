from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
from datasets import load_dataset


class SftDataset(Dataset):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    INSTRUCTION = "summarize: "

    def __init__(self, split, debug=False):
        self.dataset = load_dataset("CarperAI/openai_summarize_tldr")[split]
        if debug:
            self.dataset = self.dataset.select(range(5))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def sanity_check(self):
        prompts = [SftDataset.INSTRUCTION + self.dataset[0]["prompt"]]
        input_ids = SftDataset._tokenize(prompts)

        return input_ids, prompts[0], self.dataset[0]["label"]

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
        prompts = [SftDataset.INSTRUCTION + item["prompt"] for item in batch]
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
        dataset, batch_size=2, shuffle=True, collate_fn=SftDataset.collate_fn
    )

    for row in dataloader:
        print(row)
        break
