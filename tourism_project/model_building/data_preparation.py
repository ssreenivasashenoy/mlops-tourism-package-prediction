import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo

HF_USERNAME = "ShenoySreenivas"
RAW_DATASET = f"{HF_USERNAME}/tourism-data"
PROCESSED_DATASET = f"{HF_USERNAME}/tourism-data-processed"

processed_dir = "tourism_project/data/processed"
os.makedirs(processed_dir, exist_ok=True)

api = HfApi(token=os.getenv("HF_TOKEN"))

print(f"Loading dataset: {RAW_DATASET}")
dataset = load_dataset(RAW_DATASET, split="train")
df = dataset.to_pandas()
print("Dataset loaded.")

# Data cleaning
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

df["Gender"] = df["Gender"].replace({"FeMale": "Female"})
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

if "CustomerID" in df.columns:
    df.drop(columns=["CustomerID"], inplace=True)

df = df.fillna(df.mode().iloc[0])

# Encoding categorical columns
categorical_cols = df.select_dtypes(include="object").columns.tolist()

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Train-test split (features + target together)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["ProdTaken"]
)

# Save locally
train_df.to_csv(f"{processed_dir}/train.csv", index=False)
test_df.to_csv(f"{processed_dir}/test.csv", index=False)

print("Train and test datasets saved locally.")

# Upload to Hugging Face
try:
    api.repo_info(PROCESSED_DATASET, repo_type="dataset")
    print("Processed dataset repository exists. Uploading files.")
except:
    print("Processed dataset repository not found. Creating new dataset.")
    create_repo(repo_id=PROCESSED_DATASET, repo_type="dataset", private=False)

api.upload_file(
    path_or_fileobj=f"{processed_dir}/train.csv",
    path_in_repo="train.csv",
    repo_id=PROCESSED_DATASET,
    repo_type="dataset",
    create_pr=False
)

api.upload_file(
    path_or_fileobj=f"{processed_dir}/test.csv",
    path_in_repo="test.csv",
    repo_id=PROCESSED_DATASET,
    repo_type="dataset",
    create_pr=False
)

print(f"Processed dataset uploaded successfully: https://huggingface.co/datasets/{PROCESSED_DATASET}")
