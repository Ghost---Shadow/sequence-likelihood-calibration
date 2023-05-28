from pathlib import Path
import shutil
from tqdm import tqdm
from transformers import T5ForConditionalGeneration
from wrapped_datasets.comparison_dataset import ComparisionDataset
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter

model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.train()

EPOCHS = 3
BATCH_SIZE = 2
LR = 1e-3
DEBUG = False

train_dataset = ComparisionDataset("train", debug=DEBUG)
val_dataset = ComparisionDataset("valid1", debug=DEBUG)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=ComparisionDataset.collate_fn,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=ComparisionDataset.collate_fn,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

device = "cuda"

# Initialize TensorBoard writer
writer = SummaryWriter()
global_steps = 0

best_val_loss = 1e9
for epoch in range(EPOCHS):
    model.train().to(device)
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # Log training loss to TensorBoard
        writer.add_scalar("Loss/train", loss.item(), global_steps)
        global_steps += 1

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
        # https://stackoverflow.com/a/52784607/1217998
        prompt = prompt.replace("\n", "  \n")
        sample_input_ids = sample_input_ids.to(device)
        model.eval()
        generated = model.generate(sample_input_ids, max_length=100, temperature=0.0)
        model.train()
        generated_text = ComparisionDataset.tokenizer.decode(generated[0])
        writer.add_text(
            "Text Generation",
            f"### Input ###\n\n{prompt}\n\n### Expected ###\n\n{expected}\n\n### Output ###\n\n{generated_text}",
            epoch,
        )

        # Save the model checkpoint
        checkpoint_path = Path(f"./checkpoints/classifiers/epoch_{epoch+1}")
        try:
            shutil.rmtree(checkpoint_path)
        except Exception:
            ...
        checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(checkpoint_path.parent / "best.txt", "w") as f:
                f.write(str(epoch))
        model.save_pretrained(checkpoint_path)

# Close the writer
writer.close()
