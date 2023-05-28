from tqdm import tqdm
from train.utils import rmrf_then_mkdir
from transformers import T5ForConditionalGeneration
from wrapped_datasets.slic_dataset import SlicDataset
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.train()

EPOCHS = 10
BATCH_SIZE = 2
LR = 1e-5
DEBUG = True

jsonl_path = "generated_data/classified_summaries_length/result.jsonl"
train_dataset = SlicDataset(jsonl_path=jsonl_path, split="train", debug=DEBUG)
val_dataset = SlicDataset(jsonl_path=jsonl_path, split="valid", debug=DEBUG)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=SlicDataset.collate_fn,
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=SlicDataset.collate_fn
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

device = "cuda"
model.to(device)


def slic_loss_logits(model, batch, delta=0.5, lambda_reg=0.5):
    prompt = batch["prompts"].to(device)
    chosen = batch["chosens"].to(device)
    rejected = batch["rejecteds"].to(device)
    reference = batch["references"].to(device)

    # Pad all sequences to the same length
    max_length = max(chosen.size(1), rejected.size(1))
    chosen = F.pad(chosen, (0, max_length - chosen.size(1)))
    rejected = F.pad(rejected, (0, max_length - rejected.size(1)))

    # Calculate the log probabilities of the sequences
    log_prob_chosen = model(input_ids=prompt, labels=chosen).logits
    log_prob_rejected = model(input_ids=prompt, labels=rejected).logits
    reference_loss = model(input_ids=prompt, labels=reference).loss

    # Calculate the calibration loss
    calibration_loss = F.relu(delta - log_prob_chosen + log_prob_rejected)
    calibration_loss = calibration_loss.mean()

    # Calculate the regularization loss
    regularization_loss = lambda_reg * reference_loss

    # Combine the losses
    loss = calibration_loss + regularization_loss

    return loss


def slic_loss(model, batch, train_step=0, writer=None, delta=0.1, lambda_reg=0.1):
    prompt = batch["prompts"].to(device)
    chosen = batch["chosens"].to(device)
    rejected = batch["rejecteds"].to(device)
    reference = batch["references"].to(device)

    chosen_loss = model(input_ids=prompt, labels=chosen).loss
    rejected_loss = model(input_ids=prompt, labels=rejected).loss
    reference_loss = model(input_ids=prompt, labels=reference).loss

    if writer is not None:
        writer.add_scalar("Loss/chosen", chosen_loss, train_step)
        writer.add_scalar("Loss/rejected", rejected_loss, train_step)
        writer.add_scalar("Loss/reference", reference_loss, train_step)

    # Calculate the calibration loss
    calibration_loss = F.relu(delta + chosen_loss - rejected_loss)

    # Calculate the regularization loss
    regularization_loss = lambda_reg * reference_loss

    # Combine the losses
    loss = calibration_loss + regularization_loss
    return loss


# Initialize TensorBoard writer
# LOG_DIR = "runs/slic/long_short_logits_1"
LOG_DIR = "runs/slic/long_short_loss_1"
rmrf_then_mkdir(LOG_DIR)
writer = SummaryWriter(log_dir=LOG_DIR)

# loss_fn = slic_loss_logits
loss_fn = slic_loss

train_step = 0
for epoch in range(EPOCHS):
    total_loss = 0
    model.eval().to(device)
    with torch.no_grad():
        for batch in tqdm(val_loader):
            loss = loss_fn(model, batch)

            total_loss += loss.item()

    val_loss = total_loss / len(val_loader)
    print(f"Validation loss {val_loss}")

    # Log validation loss to TensorBoard
    writer.add_scalar("Loss/validation", val_loss, epoch)

    # Generate a sample text and log it to TensorBoard
    with torch.no_grad():
        (
            sample_input_ids,
            prompt,
            chosen,
            rejected,
            reference,
        ) = val_dataset.sanity_check(check_idx=0)
        # ) = train_dataset.sanity_check(check_idx=2)
        sample_input_ids = sample_input_ids.to(device)
        model.eval()
        generated = model.generate(sample_input_ids, max_length=100, temperature=0.0)
        model.train()
        generated_text = SlicDataset.tokenizer.decode(generated[0])
        writer.add_text(
            "Text Generation",
            f"### Input ###\n\n{prompt}\n\n### Output ###\n\n{generated_text}\n\n### Chosen ###\n\n{chosen}\n\n### Rejected ###\n\n{rejected}\n\n### Reference ###\n\n{reference}",
            epoch,
        )

    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        train_step += 1
        loss = loss_fn(model, batch, train_step, writer)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    print(f"Train loss {train_loss}")

    # Log training loss to TensorBoard
    writer.add_scalar("Loss/train", train_loss, epoch)

# Close the writer
writer.close()
