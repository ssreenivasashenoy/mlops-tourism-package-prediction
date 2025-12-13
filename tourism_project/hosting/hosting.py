import os
from huggingface_hub import HfApi, create_repo, upload_folder

HF_USERNAME = "ShenoySreenivas"
SPACE_NAME = "tourism-app"

repo_id = f"{HF_USERNAME}/{SPACE_NAME}"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=repo_id, repo_type="space")
    print(f"Space '{repo_id}' already exists.")
except:
    print(f"Creating new space {repo_id}")
    create_repo(repo_id=repo_id, repo_type="space", private=False, space_sdk="docker")

print("Uploading deployment files...")

upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=repo_id,
    repo_type="space",
)

print(f"Deployment uploaded successfully: https://huggingface.co/spaces/{repo_id}")
