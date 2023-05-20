from tqdm import tqdm
from transformers import T5ForConditionalGeneration
from sft_dataset import SftDataset
from torch.utils.data import DataLoader
import torch

model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.train()

train_dataset = SftDataset('train')
val_dataset = SftDataset('valid')

EPOCHS = 3
BATCH_SIZE = 2
LR = 1e-3

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=SftDataset.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=SftDataset.collate_fn)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

device = 'cuda'

for epoch in range(EPOCHS):
    total_loss = 0
    model.train().to(device)
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Train loss {total_loss / len(train_loader)}')

    total_loss = 0
    model.eval().to(device)
    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()

    print(f'Validation loss {total_loss / len(val_loader)}')
