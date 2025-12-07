from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Set Hugging Face dataset repo info
HF_USERNAME = "ShenoySreenivas"
DATASET_NAME = "tourism-data"
repo_id = f"{HF_USERNAME}/{DATASET_NAME}"
repo_type = "dataset"

# Path to local data folder
local_data_folder = "tourism_project/data"

# Hugging Face Token
api = HfApi(token=os.getenv("HF_TOKEN"))

# Check if dataset exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Uploading files...")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new dataset...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset '{repo_id}' created successfully.")

# Upload CSV file(s)
api.upload_folder(
    folder_path=local_data_folder,
    repo_id=repo_id,
    repo_type=repo_type,
)

print(f"Uploaded dataset files to Hugging Face: {repo_id}")
