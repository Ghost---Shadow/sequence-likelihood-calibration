from tqdm import tqdm
from transformers import T5ForConditionalGeneration
from sft_dataset import SftDataset
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter

model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.train()

EPOCHS = 3
BATCH_SIZE = 2
LR = 1e-3
DEBUG = True

train_dataset = SftDataset("train", debug=DEBUG)
val_dataset = SftDataset("valid", debug=DEBUG)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=SftDataset.collate_fn
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=SftDataset.collate_fn
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

device = "cuda"

# Initialize TensorBoard writer
writer = SummaryWriter()

for epoch in range(EPOCHS):
    total_loss = 0
    model.train().to(device)
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    print(f"Train loss {train_loss}")

    # Log training loss to TensorBoard
    writer.add_scalar("Loss/train", train_loss, epoch)

    total_loss = 0
    model.eval().to(device)
    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()

    val_loss = total_loss / len(val_loader)
    print(f"Validation loss {val_loss}")

    # Log validation loss to TensorBoard
    writer.add_scalar("Loss/validation", val_loss, epoch)

    # Generate a sample text and log it to TensorBoard
    with torch.no_grad():
        sample_input_ids, prompt, expected = val_dataset.sanity_check()
        sample_input_ids = sample_input_ids.to(device)
        generated = model.generate(sample_input_ids, max_length=100, temperature=0.0)
        generated_text = SftDataset.tokenizer.decode(generated[0])
        writer.add_text(
            "Text Generation",
            f"### Input ###\n\n{prompt}\n\n### Expected ###\n\n{expected}\n\n### Output ###\n\n{generated_text}",
            epoch,
        )

# Close the writer
writer.close()
