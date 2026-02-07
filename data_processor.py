import json
import os
import random
import glob
import re
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# ============================================================================
# CONFIGURATION - Optimized for Personality Capture
# ============================================================================
DATA_DIR = r"C:\Users\aaron\Downloads\discord data package"
OUTPUT_FILE = "train.jsonl"

# Conversation window settings (keeps context coherent)
MAX_CONTEXT_MESSAGES = 5        # Previous messages for context
MAX_TIME_GAP_MINUTES = 30       # Max gap before treating as new conversation
MIN_YOUR_MSG_LENGTH = 10        # Skip very short messages ("ok", "lol", etc.)
MIN_CONTEXT_MESSAGES = 1        # Minimum context before your response

# Quality filters
SKIP_PATTERNS = [
    r"^https?://\S+$",          # Pure URLs
    r"^<@!?\d+>$",               # Just mentions
    r"^<:\w+:\d+>$",             # Just custom emojis
    r"^\W+$",                    # Just punctuation/symbols
]

# ============================================================================


def is_valid_message(content: str) -> bool:
    """Check if message passes quality filters."""
    if not content or not content.strip():
        return False
    
    for pattern in SKIP_PATTERNS:
        if re.match(pattern, content.strip()):
            return False
    
    return True


def parse_timestamp(ts_str: str) -> datetime:
    """Parse Discord timestamp format."""
    try:
        # Handle various timestamp formats
        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z", 
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
        ]:
            try:
                return datetime.strptime(ts_str.replace("+00:00", "+0000"), fmt)
            except ValueError:
                continue
        # Fallback: just use current time if parsing fails
        return datetime.now()
    except:
        return datetime.now()


def format_conversation_example(context_msgs: list, your_response: str) -> str:
    """
    Format as a natural conversation following Llama 3 chat format.
    No system prompt needed - the model learns your style from the responses directly.
    """
    parts = []
    
    # Context as user turn (what prompted your response)
    if context_msgs:
        context_text = "\n".join([f"[{msg['author']}]: {msg['content']}" for msg in context_msgs])
        parts.append(f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{context_text}<|eot_id|>")
    else:
        # No context - just your standalone message
        parts.append(f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nContinue the conversation:<|eot_id|>")
    
    # Your response as assistant turn
    parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{your_response}<|eot_id|>")
    
    return "".join(parts)


def process_data(data_dir, output_file):
    print(f"Scanning {data_dir}...")
    print("=" * 60)
    
    messages_path = Path(data_dir) / "Messages"
    if not messages_path.exists():
        print(f"Error: Messages directory not found at {messages_path}")
        return

    # Find all messages.json files
    pattern = str(messages_path / "c*" / "messages.json")
    files = glob.glob(pattern)
    print(f"Found {len(files)} channel files.")

    # Organize messages by channel with timestamps
    channels = defaultdict(list)
    your_user_id = None  # Will detect from most common sender
    user_message_counts = defaultdict(int)

    for f_path in files:
        channel_id = Path(f_path).parent.name
        
        try:
            with open(f_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for msg in data:
                content = msg.get("Contents", "").strip()
                author = msg.get("Author", "Unknown")
                timestamp_str = msg.get("Timestamp", "")
                
                if not is_valid_message(content):
                    continue
                
                # Track user message counts to identify "you"
                user_message_counts[author] += 1
                
                channels[channel_id].append({
                    "content": content,
                    "author": author,
                    "timestamp": parse_timestamp(timestamp_str),
                    "raw_ts": timestamp_str,
                })
                
        except Exception as e:
            print(f"Error processing {f_path}: {e}")
            continue

    # Identify "you" as the most frequent author (it's your data export)
    your_user_id = max(user_message_counts, key=user_message_counts.get)
    print(f"Identified you as: {your_user_id} ({user_message_counts[your_user_id]} messages)")
    
    total_messages = sum(len(msgs) for msgs in channels.values())
    print(f"Total valid messages across channels: {total_messages}")

    # Sort each channel by timestamp
    for channel_id in channels:
        channels[channel_id].sort(key=lambda x: x["timestamp"])

    # Generate training examples: context + your response
    training_examples = []
    
    for channel_id, messages in channels.items():
        context_buffer = []
        
        for i, msg in enumerate(messages):
            # Check time gap - if too long, reset context
            if context_buffer:
                time_gap = (msg["timestamp"] - context_buffer[-1]["timestamp"])
                if time_gap > timedelta(minutes=MAX_TIME_GAP_MINUTES):
                    context_buffer = []
            
            if msg["author"] == your_user_id:
                # This is YOUR message - create training example
                if len(msg["content"]) >= MIN_YOUR_MSG_LENGTH:
                    if len(context_buffer) >= MIN_CONTEXT_MESSAGES or random.random() < 0.2:
                        # Create example with context
                        example_text = format_conversation_example(
                            context_buffer[-MAX_CONTEXT_MESSAGES:],
                            msg["content"]
                        )
                        training_examples.append({"text": example_text})
                
                # Add your message to context for following messages
                context_buffer.append({
                    "author": "You",  # Normalize your name
                    "content": msg["content"],
                    "timestamp": msg["timestamp"],
                })
            else:
                # Someone else's message - add to context
                context_buffer.append({
                    "author": msg["author"].split("#")[0][:20],  # Shorten usernames
                    "content": msg["content"],
                    "timestamp": msg["timestamp"],
                })
            
            # Limit context buffer size
            if len(context_buffer) > MAX_CONTEXT_MESSAGES * 2:
                context_buffer = context_buffer[-MAX_CONTEXT_MESSAGES:]

    print(f"\nGenerated {len(training_examples)} training examples.")
    
    # Shuffle to mix conversations from different channels
    random.shuffle(training_examples)
    
    # Remove duplicates based on content
    seen = set()
    unique_examples = []
    for ex in training_examples:
        text_hash = hash(ex["text"])
        if text_hash not in seen:
            seen.add(text_hash)
            unique_examples.append(ex)
    
    print(f"After deduplication: {len(unique_examples)} examples.")
    
    # Save to JSONL
    output_path = Path(output_file).resolve()
    with open(output_path, "w", encoding="utf-8") as f:
        for item in unique_examples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\nSaved training data to {output_path}")
    print("=" * 60)
    
    # Print a sample
    if unique_examples:
        print("\nSample training example:")
        print("-" * 40)
        sample = random.choice(unique_examples)["text"]
        print(sample[:500] + "..." if len(sample) > 500 else sample)


if __name__ == "__main__":
    process_data(DATA_DIR, OUTPUT_FILE)
