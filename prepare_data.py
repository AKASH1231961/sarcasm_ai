import pandas as pd
import json
import re

# 1. Load Kaggle CSV (please place friends_script.csv in the same directory)
df = pd.read_csv("friends_script.csv")

# 2. Basic cleaning: ensure no missing values in character names and lines
df = df.dropna(subset=['Name', 'Lines'])
df = df.reset_index(drop=True)          # 🔑 KEY FIX: make index contiguous

df['Character'] = df['Name'].astype(str).str.strip()
df['Lines'] = df['Lines'].astype(str).str.strip()

# 3. Find all indices of Chandler's lines
chandler_indices = df[df['Name'].str.contains('Chandler', case=False, na=False)].index.tolist()

# 4. Construct conversation pairs: previous line as user, Chandler's line as assistant
data = []
for idx in chandler_indices:
    if idx == 0:  # Skip the first line (no context)
        continue
    
    prev_idx = idx - 1
    user_text = df.loc[prev_idx, 'Lines']
    assistant_text = df.loc[idx, 'Lines']
    prev_speaker = df.loc[prev_idx, 'Name']
    
    # Filter: user text cannot be too short, cannot be Chandler himself, and cannot be action descriptions
    if (len(user_text) < 5 or 
        'Chandler' in prev_speaker or 
        user_text.startswith('[') or 
        user_text.startswith('(')):
        continue
    
    # Construct Llama 3.1 messages format
    entry = {
        "messages": [
            {"role": "system", "content": "You are Chandler Bing from Friends. Respond with sarcasm, wit, and self-deprecating humor. Keep responses concise and in character."},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text}
        ]
    }
    data.append(entry)

# 5. Deduplicate and save
unique_data = []
seen = set()
for item in data:
    key = item["messages"][1]["content"] + "||" + item["messages"][2]["content"]
    if key not in seen:
        seen.add(key)
        unique_data.append(item)

with open("train.jsonl", "w", encoding="utf-8") as f:
    for item in unique_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Generated {len(unique_data)} training samples, saved to train.jsonl")