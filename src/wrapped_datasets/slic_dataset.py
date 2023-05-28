import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer
from datasets import load_dataset
from wrapped_datasets.utils import clean_prompt


class SlicDataset(Dataset):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    INSTRUCTION = "summarize: "

    def __init__(self, split, jsonl_path, debug=False):
        SlicDataset.tokenizer.add_tokens(["\n"])

        # Store references from HF dataset
        hf_dataset = load_dataset("CarperAI/openai_summarize_tldr")[split]

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

        # Merge two datasets
        for i in tqdm(range(len(self.dataset))):
            sft_row = hf_dataset[i]

            prompt = self.dataset[i]["prompt"]
            prompt = clean_prompt(prompt)
            sft_prompt = sft_row["prompt"]
            sft_prompt = clean_prompt(sft_prompt)
            assert sft_prompt.startswith(prompt), sft_prompt + "\n\n" + prompt

            reference = sft_row["label"]

            self.dataset[i] = {
                **self.dataset[i],
                "reference": reference,
            }

        if debug:
            self.dataset = self.dataset[:5]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    @staticmethod
    def _tokenize(s, padding=True):
        tokens = SlicDataset.tokenizer(
            s,
            padding=padding,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        return tokens
    
    def sanity_check(self):
        prompts = [self[0]["prompt"]]
        input_ids = SlicDataset._tokenize(prompts)

        return (
            input_ids,
            self[0]["prompt"],
            self[0]["chosen"],
            self[0]["rejected"],
            self[0]["reference"],
        )

    @staticmethod
    def collate_fn(batch):
        prompts = [item["prompt"] for item in batch]
        chosens = [item["chosen"] for item in batch]
        rejecteds = [item["rejected"] for item in batch]
        references = [item["reference"] for item in batch]

        prompts = SlicDataset._tokenize(prompts)
        chosens = SlicDataset._tokenize(chosens)
        rejecteds = SlicDataset._tokenize(rejecteds)
        references = SlicDataset._tokenize(references)

        return {
            "prompts": prompts,
            "chosens": chosens,
            "rejecteds": rejecteds,
            "references": references,
        }


def test_slic_dataloader(dataloader, outfile_name, tokenizer):
    # Open a txt file
    with open(outfile_name, "w") as f:
        # Loop over the first 5 items of dataloader
        for i, row in enumerate(dataloader):
            if i >= 5:  # only need the first 5 items
                break

            # Detokenize input_ids and labels
            prompts = tokenizer.batch_decode(row["prompts"], skip_special_tokens=True)
            chosens = tokenizer.batch_decode(row["chosens"], skip_special_tokens=True)
            rejecteds = tokenizer.batch_decode(
                row["rejecteds"], skip_special_tokens=True
            )
            references = tokenizer.batch_decode(
                row["references"], skip_special_tokens=True
            )

            # Write detokenized texts to txt file
            for prompt, chosen, rejected, reference in zip(
                prompts, chosens, rejecteds, references
            ):
                f.write(f"prompt: {prompt}\n")
                f.write(f"chosen: {chosen}\n")
                f.write(f"rejected: {rejected}\n")
                f.write(f"reference: {reference}\n")
                f.write("---\n")


if __name__ == "__main__":
    jsonl_path = "generated_data/classified_summaries_length/result.jsonl"
    dataset = SlicDataset(jsonl_path=jsonl_path, split="train", debug=True)
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=SlicDataset.collate_fn
    )

    sanity_path = Path("generated_data/sanity_check")
    sanity_path.mkdir(exist_ok=True, parents=True)

    tokenizer = SlicDataset.tokenizer

    test_slic_dataloader(dataloader, sanity_path / "slic.txt", tokenizer)
