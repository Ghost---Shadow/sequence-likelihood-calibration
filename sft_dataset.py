from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
from datasets import load_dataset

class SftDataset(Dataset):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    def __init__(self, split, debug=False):
        # dataset = load_dataset("CarperAI/openai_summarize_comparisons")
        self.dataset = load_dataset("CarperAI/openai_summarize_tldr")[split]
        if debug:
            self.dataset = self.dataset.select(range(100))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    @staticmethod
    def collate_fn(batch):
        INSTRUCTION = 'summarize: '

        prompts = [INSTRUCTION + item['prompt'] for item in batch]
        labels = [item['label'] for item in batch]

        input_ids = SftDataset.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").input_ids
        labels = SftDataset.tokenizer(labels, padding=True, truncation=True, return_tensors="pt").input_ids

        return {
            'input_ids': input_ids,
            'labels': labels,
        }

if __name__ == "__main__":
    dataset = SftDataset(split='train')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=SftDataset.collate_fn)

    for row in dataloader:
        print(row)
        break
