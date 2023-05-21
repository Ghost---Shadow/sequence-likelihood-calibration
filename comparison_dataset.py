from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
from datasets import load_dataset


class ComparisionDataset(Dataset):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    INSTRUCTION = "summarize: "

    def __init__(self, split, debug=False):
        self.dataset = load_dataset("CarperAI/openai_summarize_comparisons")[split]
        if debug:
            self.dataset = self.dataset.select(range(5))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def sanity_check(self):
        prompts = [ComparisionDataset.INSTRUCTION + self.dataset[0]["prompt"]]
        input_ids = ComparisionDataset._tokenize(prompts)

        return (
            input_ids,
            prompts[0],
            self.dataset[0]["chosen"],
            self.dataset[0]["rejected"],
        )

    @staticmethod
    def _tokenize(s):
        tokens = ComparisionDataset.tokenizer(
            s,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        return tokens

    @staticmethod
    def collate_fn(batch):
        prompts = [ComparisionDataset.INSTRUCTION + item["prompt"] for item in batch]
        chosen_labels = [item["chosen"] for item in batch]
        rejected_labels = [item["rejected"] for item in batch]

        input_ids = ComparisionDataset._tokenize(prompts)
        chosen_labels = ComparisionDataset._tokenize(chosen_labels)
        rejected_labels = ComparisionDataset._tokenize(rejected_labels)

        return {
            "input_ids": input_ids,
            "chosen_labels": chosen_labels,
            "rejected_labels": rejected_labels,
        }


if __name__ == "__main__":
    dataset = ComparisionDataset(split="train")
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=ComparisionDataset.collate_fn
    )

    for row in dataloader:
        print(row)
        break
