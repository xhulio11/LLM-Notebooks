from functions import * 
################################################################################
# transformers parameters
################################################################################

# The pre-trained model from the Hugging Face Hub to load and fine-tune
model_name = "meta-llama/Llama-3.2-1B"

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
load_in_4bit = True

# Activate nested quantization for 4-bit base models (double quantization)
bnb_4bit_use_double_quant = True

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Compute data type for 4-bit base models
bnb_4bit_compute_dtype = torch.bfloat16

# Load model from Hugging Face Hub with model name and bitsandbytes configuration
print('/n/n')
bnb_config = create_bnb_config(load_in_4bit, bnb_4bit_use_double_quant, bnb_4bit_quant_type, bnb_4bit_compute_dtype)
model, tokenizer = load_model(model_name, bnb_config)

"""LOAD DATASET"""
dataset = load_from_disk('/home/t/tzelilai/Desktop/Thesis/Llama-3.2-1B/articles_dataset')

# Split Data
# train_dataset = dataset["train"].select(range(50))
# eval_dataset = dataset["eval"].select(range(10))
train_dataset = dataset["train"]
eval_dataset = dataset["eval"]

print(f"Train dataset size: {len(train_dataset)}")

"""Determing padding for lower sequences"""

# Suppose you're using a LLaMA or GPT-like tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# Preprocess
def preprocess_data(examples):
    tokenized = tokenizer(
        examples["content"],
        truncation=True,
        #return_overflowing_tokens=True,
        max_length=8192,
        stride=0
    )
    return tokenized

train_dataset = train_dataset.map(preprocess_data, batched=True)
eval_dataset = eval_dataset.map(preprocess_data, batched=True)

# Select a small sample for testing 
# train_dataset = train_dataset.select(range(100))
# eval_dataset = eval_dataset.select(range(10))

print("Set the labels :[input_ids, attention_mask, labels]")
# Set the format of dataset 
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

print("Remove columng 'content' ")
train_dataset = train_dataset.remove_columns(["content"])
eval_dataset = eval_dataset.remove_columns(["content"])

# Take only 10 examples from the dataset
# small_train_dataset = train_dataset.select(range(10))
# small_eval_dataset = eval_dataset.select(range(10))

"""Prepare Training"""
################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 16

# Alpha parameter for LoRA scaling
lora_alpha = 64

# Dropout probability for LoRA layers
lora_dropout = 0.1

# Bias
bias = "none"

# Task type
task_type = "SEQ_CLS"

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "/home/t/tzelilai/Desktop/Thesis/results-2"

# Batch size per GPU for training
per_device_train_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 4

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Optimizer to use
optim = "paged_adamw_32bit"

# Number of training steps (overrides num_train_epochs)
# max_steps = 20

# Number of epochs 
num_train_epochs = 3

# Linear warmup steps from 0 to learning_rate
warmup_steps = 2

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = True

# Log every X updates steps
logging_steps = 1


"""
-----------------------------Train-------------------------------
"""
print("Training Started")

fine_tune(model, tokenizer, train_dataset,eval_dataset, lora_r, lora_alpha, lora_dropout, 
          bias, task_type, per_device_train_batch_size, gradient_accumulation_steps, 
          warmup_steps, num_train_epochs, learning_rate, fp16, logging_steps, output_dir, optim)

print("Training Ended")


