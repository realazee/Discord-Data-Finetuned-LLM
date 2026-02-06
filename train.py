import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ============================================================================
# BASIC CONFIGURATION - TinyLlama 1.1B
# ============================================================================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_SEQ_LENGTH = 512  
TRAIN_FILE = "train.jsonl"
OUTPUT_DIR = "./fine_tuned_discord_model"

# Optimized Training Params for RTX 5090 - azee's beastly GPU xd
EPOCHS = 1          
BATCH_SIZE = 32     
GRADIENT_ACC = 1    
LEARNING_RATE = 2e-4
MAX_SAMPLES = 25000 
# ============================================================================

def main():
    print(f"Loading model: {MODEL_NAME}")
    
    # 1. Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # 2. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "right" 

    # 4. Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Load & Tokenize Dataset
    dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
    
    # Optimization: Use only the most recent messages if possible, or just shuffle and take subset
    # Assuming jsonl order is roughly chronological or we just want a sample.
    if len(dataset) > MAX_SAMPLES:
        print(f"Dataset has {len(dataset)} examples. Limiting to {MAX_SAMPLES} to speed up training.")
        dataset = dataset.shuffle(seed=42).select(range(MAX_SAMPLES))
    
    def tokenize_function(examples):
        # Simply tokenize the raw text since we are doing causal LM on pre-grouped data
        return tokenizer(examples["text"], truncation=True, max_length=MAX_SEQ_LENGTH)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # 6. Trainer
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACC,
            learning_rate=LEARNING_RATE,
            logging_steps=10,
            num_train_epochs=EPOCHS,
            optim="paged_adamw_8bit",
            fp16=True,
            save_strategy="epoch",
        ),
    )

    # 7. Train
    print("Starting training...")
    trainer.train()
    
    # 8. Save
    print("Saving model...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
