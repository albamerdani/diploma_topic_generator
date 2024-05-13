from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load tokenizer and model
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Prepare dataset
train_path = 'thesis_topics.txt'
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_path,
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Setup training arguments
training_args = TrainingArguments(
    output_dir='/content/results',          # output directory
    overwrite_output_dir=False,       # overwrite the content of the output directory
    num_train_epochs=4,              # number of training epochs
    per_device_train_batch_size=4,   # batch size for training
    save_steps=10_000,               # number of updates steps before saving model
    save_total_limit=2,              # number of total save model.
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./thesis_topic_generator')
tokenizer.save_pretrained('./thesis_topic_generator')

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('./thesis_topic_generator')
tokenizer = GPT2Tokenizer.from_pretrained('./thesis_topic_generator')

# Function to generate text
def generate_text(prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Example usage
print(generate_text("Deep learning and computer vision"))