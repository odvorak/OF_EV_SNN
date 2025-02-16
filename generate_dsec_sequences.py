import pandas as pd
import os
import numpy as np

# Define root directory
events_path = r"E:\DSEC\saved_flow_data\event_tensors\11frames"

# Get all files in the directory
files = os.listdir(events_path)

# Sort files to maintain order
files = np.sort(files)

# Split into 80% train and 20% validation
num_files = len(files)
split_idx = int(0.8 * num_files)

train_files = files[:split_idx]
valid_files = files[split_idx:]

# Create file pairs (f1, f2) for training and validation
train_pairs = [(train_files[i], train_files[i+1]) for i in range(len(train_files) - 1)]
valid_pairs = [(valid_files[i], valid_files[i+1]) for i in range(len(valid_files) - 1)]

# Convert to DataFrame
df_train = pd.DataFrame(train_pairs, columns=['f1', 'f2'])
df_valid = pd.DataFrame(valid_pairs, columns=['f1', 'f2'])

# Define save path
save_path = r"E:\DSEC\saved_flow_data\sequence_lists"
os.makedirs(save_path, exist_ok=True)

# Save CSV files
df_train.to_csv(os.path.join(save_path, "train_split_seq.csv"), header=None, index=None)
df_valid.to_csv(os.path.join(save_path, "valid_split_seq.csv"), header=None, index=None)

print(f"Train pairs: {len(df_train)}, Validation pairs: {len(df_valid)}")
print("Train and validation splits saved successfully.")
