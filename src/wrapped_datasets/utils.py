import re


def clean_prompt(prompt, truncate_to=1024):
    if "POST:" in prompt:
        prompt = prompt.split("POST:")[1]
    prompt = prompt.replace("\\r\\n", " ")
    prompt = prompt.replace("\\n", " ")
    prompt = prompt.replace("\r\n", " ")
    prompt = prompt.replace("\n", " ")
    prompt = prompt.replace("TL;DR:", "")
    prompt = prompt.replace("summarize:", "")
    prompt = prompt.replace("<extra_id_-1>", "")
    prompt = re.sub(r"\s+", " ", prompt)

    # TODO: Do it properly
    if truncate_to is not None:
        prompt = prompt.strip()[:truncate_to].strip()

    return prompt


def test_dataloader(dataloader, outfile_name, tokenizer):
    # Open a txt file
    with open(outfile_name, "w") as f:
        # Loop over the first 5 items of dataloader
        for i, row in enumerate(dataloader):
            if i >= 5:  # only need the first 5 items
                break

            # Detokenize input_ids and labels
            input_texts = tokenizer.batch_decode(
                row["input_ids"], skip_special_tokens=True
            )
            label_texts = tokenizer.batch_decode(
                row["labels"], skip_special_tokens=True
            )

            # Write detokenized texts to txt file
            for input_text, label_text in zip(input_texts, label_texts):
                f.write(f"{input_text}{label_text}\n")
                f.write("---\n")
