import torch
from tqdm import tqdm
from wrapped_datasets.comparison_dataset import ComparisionDataset
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration


DEBUG = False
BATCH_SIZE = 1
MODEL = "t5-small"
val_dataset = ComparisionDataset("valid1", DEBUG)

model = T5ForConditionalGeneration.from_pretrained(MODEL)
model.eval()

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
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
            labels = batch["labels"].to(device)[:,:1]

            # Get the model outputs
            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits
            logits = logits[:,0,:].squeeze(axis=1)

            probs = torch.exp(logits)

            # Mask out non-relevant tokens
            mask = torch.zeros_like(logits).to(device)
            mask[:,tokenized_labels] = 1

            # Mask the logits with the tokenized labels
            masked_probs = probs * mask

            # Get the predicted labels
            predicted_labels = torch.argmax(masked_probs, dim=-1)

            # Calculate the number of correct predictions
            corrects += (predicted_labels == labels).sum().item()
            samples_seen += labels.size(0)

            pbar.update(1)
            pbar.set_description("{:.2f}".format(corrects / samples_seen))
