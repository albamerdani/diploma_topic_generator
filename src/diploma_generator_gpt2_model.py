from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import matplotlib.pyplot as plt
import pandas as pd
import os
import tensorflow as tf
from transformers import TrainerCallback

dataset_path_file = os.path.join("dataset", "diploma.xlsx")
# Step 1: Load data
df = pd.read_excel(dataset_path_file)  # Replace with your actual file path

thesis_file = os.path.join("temp", "thesis_topics.txt")
# Concatenate all thesis topics into a single text file
with open(thesis_file, 'w', encoding='utf-8') as file:
    for topic in df['theme']:
        file.write(topic + "\n")


class HistoryCallback(TrainerCallback):
    "A custom callback that records loss and other metrics."
    def __init__(self):
        self.history = {'loss': [], 'perplexity': []}

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Logs might include loss and other metrics like learning rate
        if 'loss' in logs:
            self.history['loss'].append(logs['loss'])
        if 'perplexity' in logs:  # Ensure your perplexity metric is logged appropriately
            self.history['perplexity'].append(logs['perplexity'])

# Initialize the callback
history = HistoryCallback()

def perplexity(labels, logits):
    return tf.exp(tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)))


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
    output_dir='/content',          # output directory
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
    compute_metrics=perplexity,
    callbacks=[history]
)

# Train the model
trainer.train()
trainer.save_model("./gpt2_finetuned")

# Now you can access the history similar to Keras
print(history.history['loss'])
print(history.history['perplexity'])

# Assume 'history' has the perplexity values over epochs or training steps
#plt.plot(history['perplexity'])
#plt.title('Model Perplexity')
#plt.ylabel('Perplexity')
#plt.xlabel('Epoch')
#plt.legend(['Train'], loc='upper left')
#plt.show()

# Save the model
model.save_pretrained('./thesis_topic_generator')
tokenizer.save_pretrained('./thesis_topic_generator')


# Function to generate text
def generate_text(prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Example usage
print(generate_text("Wireless network"))

# Example usage
print(generate_text("Deep learning and computer vision"))