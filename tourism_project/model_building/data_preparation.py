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

# Load dataset
print(f"Loading dataset: {RAW_DATASET}")
dataset = load_dataset(RAW_DATASET, split="train")
df = dataset.to_pandas()
print("Dataset loaded.")

# Cleaning
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

df["Gender"] = df["Gender"].replace({"FeMale": "Female"})
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

if "CustomerID" in df.columns:
    df.drop(columns=["CustomerID"], inplace=True)

df = df.fillna(df.mode().iloc[0])

# Encoding categorical columns
categorical_cols = df.select_dtypes(include="object").columns.tolist()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = list(le.classes_)

# Train/Test split
X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save files
X_train.to_csv(f"{processed_dir}/X_train.csv", index=False)
X_test.to_csv(f"{processed_dir}/X_test.csv", index=False)
y_train.to_csv(f"{processed_dir}/y_train.csv", index=False)
y_test.to_csv(f"{processed_dir}/y_test.csv", index=False)

print("Train-test files saved locally.")

# Upload to Hugging Face
try:
    api.repo_info(PROCESSED_DATASET, repo_type="dataset")
    print("Repository exists, uploading files.")
except:
    print("Dataset not found. Creating dataset.")
    create_repo(repo_id=PROCESSED_DATASET, repo_type="dataset", private=False)

for file in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
    api.upload_file(
        path_or_fileobj=f"{processed_dir}/{file}",
        path_in_repo=file,
        repo_id=PROCESSED_DATASET,
        repo_type="dataset",
        create_pr=False
    )

print(f"Dataset uploaded successfully: https://huggingface.co/datasets/{PROCESSED_DATASET}")
