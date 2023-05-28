import argparse
from pathlib import Path
import time
from tqdm import tqdm
from train.utils import rmrf_then_mkdir
from transformers import T5ForConditionalGeneration
from wrapped_datasets.slic_dataset import SlicDataset
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="t5-small",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--train-jsonl-path",
        type=str,
        default="./generated_data/classified_summaries_length/train.jsonl",
    )
    parser.add_argument(
        "--val-jsonl-path",
        type=str,
        default="./generated_data/classified_summaries_length/valid.jsonl",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="slic_loss",
        choices=["slic_loss", "slic_loss_logits"],
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="./runs/slic/long_short",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/slic/long_short",
    )
    parser.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()
    return args


def slic_loss_logits(
    model, batch, train_step=0, writer=None, delta=0.5, lambda_reg=0.5
):
    device = model.device
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

    if writer is not None:
        writer.add_scalar("Loss/calibration", calibration_loss, train_step)
        writer.add_scalar("Loss/reference", reference_loss, train_step)

    # Calculate the regularization loss
    regularization_loss = lambda_reg * reference_loss

    # Combine the losses
    loss = calibration_loss + regularization_loss

    return loss


def slic_loss(model, batch, train_step=0, writer=None, delta=0.5, lambda_reg=0.5):
    device = model.device
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


def setup_datasets(args):
    batch_size = args.batch_size
    train_jsonl_path = args.train_jsonl_path
    val_jsonl_path = args.val_jsonl_path
    limit = args.limit

    train_dataset = SlicDataset(jsonl_path=train_jsonl_path, split="train", limit=limit)
    val_dataset = SlicDataset(jsonl_path=val_jsonl_path, split="valid", limit=limit)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=SlicDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=SlicDataset.collate_fn,
    )
    sanity_check = val_dataset.sanity_check(check_idx=0)
    # sanity_check = train_dataset.sanity_check(check_idx=2)

    return train_loader, val_loader, sanity_check


def train_loop(args):
    model_name = args.model_name
    epochs = args.epochs
    learning_rate = args.learning_rate
    loss_type = args.loss_type
    logdir = Path(args.logdir)
    checkpoint_dir = Path(args.checkpoint_dir)

    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.train()

    train_loader, val_loader, sanity_check = setup_datasets(args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    device = "cuda"
    model.to(device)

    epoch_time = int(time.time())
    experiment_name = f"{loss_type}_{learning_rate}_{epoch_time}"
    full_logdir = logdir / experiment_name

    # Initialize TensorBoard writer
    rmrf_then_mkdir(full_logdir)
    writer = SummaryWriter(log_dir=full_logdir)

    # Checkpoints
    full_checkpoint_dir = checkpoint_dir / experiment_name
    rmrf_then_mkdir(full_checkpoint_dir)

    loss_fn = {
        "slic_loss": slic_loss,
        "slic_loss_logits": slic_loss_logits,
    }[loss_type]

    train_step = 0
    for epoch in range(epochs):
        total_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader):
                loss = loss_fn(model=model, batch=batch)

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
            ) = sanity_check
            sample_input_ids = sample_input_ids.to(device)
            model.eval()
            generated = model.generate(
                sample_input_ids, max_length=100, temperature=0.0
            )
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
            loss = loss_fn(
                model=model,
                batch=batch,
                train_step=train_step,
                writer=writer,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        print(f"Train loss {train_loss}")

        # Log training loss to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)

        # Drop one checkpoint per epoch
        torch.save(model.state_dict(), full_checkpoint_dir / f'epoch_{epoch+1}.pth')

    # Close the writer
    writer.close()


if __name__ == "__main__":
    args = parse_args()

    train_loop(args)
