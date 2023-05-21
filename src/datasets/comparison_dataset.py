import random
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
from datasets import load_dataset


class ComparisionDataset(Dataset):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    INSTRUCTION = "summarize: "

    def __init__(self, split, debug=False):
        ComparisionDataset.tokenizer.add_tokens(["\n"])
        self.dataset = load_dataset("CarperAI/openai_summarize_comparisons")[split]

        # self.dataset = self.dataset.filter(ComparisionDataset.filter_function)

        if debug:
            self.dataset = self.dataset.select(range(100))

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
    def format_row(row):
        labels = ["A", "B"]
        row["chosen"] = row["chosen"].replace("TL;DR: ", "").strip()
        row["rejected"] = row["rejected"].replace("TL;DR: ", "").strip()
        answers = [None, None]
        prompt = row["prompt"]

        prompt = prompt.split("POST: ")[1]
        prompt = prompt.replace("\\r\\n", " ")
        prompt = prompt.replace("\\n", " ")
        prompt = prompt.replace("\r\n", " ")
        prompt = prompt.replace("\n", " ")

        correct_index = random.randint(0, 1)

        correct_answer = labels[correct_index]
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


def test_dataloader(dataloader):
    # Open a txt file
    with open("dataloader_output.txt", "w") as f:
        # Loop over the first 5 items of dataloader
        for i, row in enumerate(dataloader):
            if i >= 5:  # only need the first 5 items
                break

            # Detokenize input_ids and labels
            input_texts = ComparisionDataset.tokenizer.batch_decode(
                row["input_ids"], skip_special_tokens=True
            )
            label_texts = ComparisionDataset.tokenizer.batch_decode(
                row["labels"], skip_special_tokens=True
            )

            # Write detokenized texts to txt file
            for input_text, label_text in zip(input_texts, label_texts):
                f.write(f"{input_text}{label_text}\n")
                f.write("---\n")


if __name__ == "__main__":
    dataset = ComparisionDataset(split="train", debug=True)
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=ComparisionDataset.collate_fn
    )

    test_dataloader(dataloader)