import random
import torch
from tqdm import tqdm
from wrapped_datasets.comparison_dataset import ComparisionDataset
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration


BATCH_SIZE = 10
MODEL = "t5-small"

random.seed(42)

val_dataset = ComparisionDataset("valid1", limit=100)

model = T5ForConditionalGeneration.from_pretrained(MODEL)
model.load_state_dict(
    torch.load(
        "checkpoints/classifier/tldr_comparison/t5-small_0.0001_1685346085/epoch_5.pth"
    )
)
model.eval()

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=ComparisionDataset.collate_fn,
)

device = "cuda"
model.to(device)

# Get the tokenized labels from the dataset
tokenized_labels = ComparisionDataset.tokenized_labels().to(device)
samples_seen = 0
corrects = 0


with torch.no_grad():
    with tqdm(total=len(val_loader)) as pbar:
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)[:, :1]

            # Get the model outputs
            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits

            # Squeeze
            logits = logits[:, 0, :].squeeze(axis=1)
            labels = labels.squeeze()

            probs = torch.exp(logits)

            # Mask out non-relevant tokens
            mask = torch.zeros_like(logits).to(device)
            mask[:, tokenized_labels] = 1

            # Mask the logits with the tokenized labels
            masked_probs = probs * mask

            # Get the predicted labels
            predicted_labels = torch.argmax(masked_probs, dim=-1)

            # Calculate the number of correct predictions
            assert predicted_labels.shape == labels.shape
            corrects += (predicted_labels == labels).sum().item()
            samples_seen += len(batch["labels"])

            pbar.update(1)
            pbar.set_description("{:.2f}".format(corrects / samples_seen))
