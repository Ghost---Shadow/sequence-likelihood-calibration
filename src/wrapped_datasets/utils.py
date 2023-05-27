def clean_prompt(prompt):
    prompt = prompt.split("POST: ")[1]
    prompt = prompt.replace("\\r\\n", " ")
    prompt = prompt.replace("\\n", " ")
    prompt = prompt.replace("\r\n", " ")
    prompt = prompt.replace("\n", " ")
    prompt = prompt.replace("TL;DR:", "")

    # TODO: Do it properly
    prompt = prompt.strip()[:1024].strip()

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
