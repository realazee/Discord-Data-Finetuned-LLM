import json
import os
import random
import glob
from pathlib import Path

# Configuration
DATA_DIR = r"" #enter it here, ill move it to .env later
OUTPUT_FILE = "train.jsonl"
MIN_GROUP_SIZE = 5   # Minimum messages per group
MAX_GROUP_SIZE = 10  # Maximum messages per group

def process_data(data_dir, output_file):
    print(f"Scanning {data_dir}...")
    
    messages_path = Path(data_dir) / "Messages"
    if not messages_path.exists():
        print(f"Error: Messages directory not found at {messages_path}")
        return

    # Find all messages.json files
    pattern = str(messages_path / "c*" / "messages.json")
    files = glob.glob(pattern)
    print(f"Found {len(files)} message files.")

    all_messages = []

    for f_path in files:
        try:
            with open(f_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for msg in data:
                content = msg.get("Contents", "").strip()
                
                # Basic cleaning
                if not content:
                    continue
                
                # Skip if it looks like just a URL
                if content.startswith("http://") or content.startswith("https://"):
                    if " " not in content:
                        continue
                
                all_messages.append(content)
                
        except Exception as e:
            print(f"Error processing {f_path}: {e}")
            continue

    print(f"Extracted {len(all_messages)} messages.")
    
    # Remove duplicate messages
    unique_messages = list(dict.fromkeys(all_messages))
    print(f"Unique messages: {len(unique_messages)}")
    
    # Group messages together (5-10 per group)
    # This creates more coherent training examples for raw text training
    grouped_data = []
    i = 0
    while i < len(unique_messages):
        group_size = random.randint(MIN_GROUP_SIZE, MAX_GROUP_SIZE)
        group = unique_messages[i:i + group_size]
        
        # Join messages with newlines
        grouped_text = "\n".join(group)
        grouped_data.append({"text": grouped_text})
        
        i += group_size
    
    print(f"Created {len(grouped_data)} grouped training examples.")
    
    # Save to JSONL (raw text format, no instruction)
    output_path = Path(output_file).resolve()
    with open(output_path, "w", encoding="utf-8") as f:
        for item in grouped_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Saved training data to {output_path}")

if __name__ == "__main__":
    process_data(DATA_DIR, OUTPUT_FILE)
