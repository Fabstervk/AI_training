from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set padding token (GPT-2 doesn't have a padding token by default)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Step 1: Read the CSV file directly and extract the relevant columns
file_path = 'path_to_your_file.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Assume your columns are named 'input_text' and 'output_text'
df['text'] = df['input_text'] + tokenizer.eos_token + df['output_text']

# Step 2: Convert the DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(df[['text']])

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Step 3: Set up the Trainer
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    evaluation_strategy="epoch",
    save_total_limit=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

# Step 4: Train the model
trainer.train()

# Step 5: Save the fine-tuned model
model.save_pretrained("./fine-tuned-gpt2")
tokenizer.save_pretrained("./fine-tuned-gpt2")
