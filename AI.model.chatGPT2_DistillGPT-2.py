from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can also use "distilgpt2" for a smaller model
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set padding token (GPT-2 doesn't have a padding token by default)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Example input text
input_text  = "Based on the following job description, create a cover letter applying for the role. "
"Job Description: In this role, you will oversee the controlling function for the company's Swedish operations, "
"focusing on analysis, profitability, and business control. Key tasks include analyzing and following up on product margins, profitability, KPIs, "
"and statistics, building and leading analysis and reporting functions for monitoring, developing budgets, forecasts, and financial analyses, "
"analyzing financial statements and preparing strategic decision-support material, providing financial expertise for business cases and calculations, "
"leading process development and efficiency improvements, collaborating with managers across business areas, and handling ad hoc analyses. "
"Requirements include an academic background in finance, IT/systems, or related fields, practical experience in business control and a solid understanding of accounting, "
"strong skills in Power BI, Excel, and basic SQL, proficiency in Swedish and English, and strong communication skills with a proactive, solution-oriented approach."


# Tokenize the input text with attention mask
inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)

# Generate text using the model
outputs = model.generate(
    inputs['input_ids'],
    attention_mask=inputs['attention_mask'],  # Pass the attention mask
    max_length=200,  # Maximum length of the generated text
    num_return_sequences=1,  # Number of sequences to generate
    temperature=0.7,  # Sampling temperature (lower is more deterministic)
    top_p=0.9,  # Nucleus sampling: consider tokens with cumulative probability of 0.9
    top_k=50,  # Top-k sampling: consider top 50 tokens for each step
    do_sample=True,  # Use sampling instead of greedy decoding
)

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
