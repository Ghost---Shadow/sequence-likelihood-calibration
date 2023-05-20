from transformers import T5Tokenizer, T5ForConditionalGeneration, GenerationConfig
from datasets import load_dataset

dataset = load_dataset("CarperAI/openai_summarize_comparisons")
row = dataset['train'][0]
prompt = row['prompt']

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

input_ids = tokenizer(f"summarize: {prompt}", return_tensors="pt").input_ids
generation_config = GenerationConfig(max_length=1024)

outputs = model.generate(input_ids, generation_config)

print(prompt)
print('---')
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print('---')
print(len(outputs[0]))
