import torch
from tqdm import tqdm
from datasets.comparison_dataset import ComparisionDataset
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

samples_seen = 0
corrects = 0
with torch.no_grad():
    with tqdm(total=len(val_loader)) as pbar:
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)

            chosen_loss = model(input_ids=input_ids, labels=chosen_labels).loss
            rejected_loss = model(input_ids=input_ids, labels=rejected_labels).loss

            samples_seen += 1
            if rejected_loss > chosen_loss:
                corrects += 1

            pbar.update(1)
            pbar.set_description("{:.2f}".format(corrects / samples_seen))
