import argparse
from pathlib import Path
import random
import time
from tqdm import tqdm
from train.utils import rmrf_then_mkdir
from transformers import T5ForConditionalGeneration
from wrapped_datasets.comparison_dataset import ComparisionDataset
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter


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
        "--logdir",
        type=str,
        default="./runs/classifier/tldr_comparison",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/classifier/tldr_comparison",
    )
    parser.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()
    return args


def setup_datasets(args):
    batch_size = args.batch_size
    limit = args.limit

    train_dataset = ComparisionDataset("train", limit=limit)
    val_dataset = ComparisionDataset("valid1", limit=limit)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ComparisionDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ComparisionDataset.collate_fn,
    )
    sanity_check = val_dataset.sanity_check(check_idx=0)
    # sanity_check = train_dataset.sanity_check(check_idx=2)

    return train_loader, val_loader, sanity_check


def train_loop(args):
    model_name = args.model_name
    epochs = args.epochs
    learning_rate = args.learning_rate
    logdir = Path(args.logdir)
    checkpoint_dir = Path(args.checkpoint_dir)

    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.train()

    train_loader, val_loader, sanity_check = setup_datasets(args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    device = "cuda"
    model.to(device)

    epoch_time = int(time.time())
    experiment_name = f"{model_name}_{learning_rate}_{epoch_time}"
    full_logdir = logdir / experiment_name

    # Initialize TensorBoard writer
    rmrf_then_mkdir(full_logdir)
    writer = SummaryWriter(log_dir=full_logdir)

    # Checkpoints
    full_checkpoint_dir = checkpoint_dir / experiment_name
    rmrf_then_mkdir(full_checkpoint_dir)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=full_logdir)
    global_steps = 0

    for epoch in range(epochs):
        # Compute validation loss
        model.eval()
        total_loss = 0
        total_correct = 0
        total_count = 0
        with torch.no_grad():
            random.seed(42 + epoch)
            for batch in tqdm(val_loader, desc="validation"):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

                total_loss += loss.item()

                # Calculate accuracy
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct = (predictions[0] == labels[0]).float().sum().item()
                total_correct += correct
                total_count += len(batch["labels"])

        val_loss = total_loss / total_count
        val_accuracy = total_correct / total_count
        print(f"Validation loss {val_loss}")
        print(f"Validation accuracy {val_accuracy}")

        # Log validation loss and accuracy to TensorBoard
        writer.add_scalar("Loss/validation", val_loss, epoch)
        writer.add_scalar("Accuracy/validation", val_accuracy, epoch)

        # Generate a sample text and log it to TensorBoard
        model.eval()
        with torch.no_grad():
            sample_input_ids, prompt, expected = sanity_check
            # https://stackoverflow.com/a/52784607/1217998
            prompt = prompt.replace("\n", "  \n")
            sample_input_ids = sample_input_ids.to(device)
            generated = model.generate(
                sample_input_ids, max_length=100, temperature=0.0
            )
            generated_text = ComparisionDataset.tokenizer.decode(generated[0])
            writer.add_text(
                "Text Generation",
                f"### Input ###\n\n{prompt}\n\n### Expected ###\n\n{expected}\n\n### Output ###\n\n{generated_text}",
                epoch,
            )

        model.train()
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

        # Drop one checkpoint per epoch
        torch.save(model.state_dict(), full_checkpoint_dir / f"epoch_{epoch+1}.pth")

    # Close the writer
    writer.close()


if __name__ == "__main__":
    args = parse_args()

    train_loop(args)
