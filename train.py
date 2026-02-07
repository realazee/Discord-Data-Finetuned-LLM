import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ============================================================================
# CONFIGURATION - Llama 3.1 8B Instruct (Anti-Overfitting Setup)
# ============================================================================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 1024  # Longer context for conversational data
TRAIN_FILE = "train.jsonl"
OUTPUT_DIR = "./fine_tuned_discord_model"

EPOCHS = 1                  # More epochs with early stopping
BATCH_SIZE = 16              # Smaller batch = more gradient noise (regularization)
GRADIENT_ACC = 8            # Effective batch size = 32, but with noise benefit
LEARNING_RATE = 5e-5        # Lower LR prevents overfitting
WARMUP_RATIO = 0.1          # Gradual warmup prevents early overfitting
WEIGHT_DECAY = 0.01         # L2 regularization
MAX_SAMPLES = 20000          
VALIDATION_SPLIT = 0.1      # 10% for validation to monitor overfitting
# ============================================================================

def main():
    print(f"Loading model: {MODEL_NAME}")
    print("=" * 60)
    
    # 1. Quantization Config (4-bit for memory efficiency)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 better for Llama 3
        bnb_4bit_use_double_quant=True,         # Nested quantization for memory
    )

    # 2. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",  # Use efficient attention
    )
    
    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "right" 

    # 4. Prepare for LoRA with anti-overfitting config
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=32,                   # Higher rank for 8B model capacity
        lora_alpha=64,          # Alpha = 2x rank is a good rule
        lora_dropout=0.1,       # Higher dropout to prevent overfitting
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj"       # MLP layers too
        ]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Load & Split Dataset
    print("\nLoading dataset...")
    dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
    print(f"Total examples: {len(dataset)}")
    
    # Limit samples if specified
    if MAX_SAMPLES and len(dataset) > MAX_SAMPLES:
        print(f"Limiting to {MAX_SAMPLES} examples.")
        dataset = dataset.shuffle(seed=42).select(range(MAX_SAMPLES))
    
    # Train/Validation split to monitor overfitting
    split = dataset.train_test_split(test_size=VALIDATION_SPLIT, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Validation: {len(eval_dataset)}")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH,
            padding=False,  # Dynamic padding is more efficient
        )

    print("Tokenizing datasets...")
    train_tokenized = train_dataset.map(
        tokenize_function, batched=True, remove_columns=train_dataset.column_names
    )
    eval_tokenized = eval_dataset.map(
        tokenize_function, batched=True, remove_columns=eval_dataset.column_names
    )

    # 6. Trainer with anti-overfitting config
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        
        # Batch config
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACC,
        
        # Learning rate with warmup and cosine decay
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        
        # Regularization
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=1.0,  # Gradient clipping
        
        # Training length
        num_train_epochs=EPOCHS,
        
        # Evaluation & early stopping
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Precision & optimization
        bf16=True,  # bfloat16 for Llama 3
        optim="paged_adamw_8bit",
        
        # Logging
        logging_steps=25,
        report_to="none",
        
        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,  # Stop if no improvement for 3 evals
                early_stopping_threshold=0.01,
            )
        ],
    )

    # 7. Train
    print("\n" + "=" * 60)
    print("Starting training with anti-overfitting measures:")
    print(f"  - Lower learning rate: {LEARNING_RATE}")
    print(f"  - Cosine LR schedule with {WARMUP_RATIO*100}% warmup")
    print(f"  - Weight decay: {WEIGHT_DECAY}")
    print(f"  - LoRA dropout: 0.1")
    print(f"  - Early stopping (patience=3)")
    print(f"  - Validation monitoring every 100 steps")
    print("=" * 60 + "\n")
    
    trainer.train()
    
    # 8. Save best model
    print("\nSaving best model...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
