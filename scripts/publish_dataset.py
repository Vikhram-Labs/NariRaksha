import os
import argparse
from huggingface_hub import HfApi, create_repo
from loguru import logger

def publish_dataset(dataset_dir: str, repo_id: str, token: str):
    logger.info(f"Publishing dataset from {dataset_dir} to {repo_id}")
    api = HfApi(token=token)
    
    # Create repo if not exists
    create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)
    
    # Upload files
    api.upload_folder(
        folder_path=dataset_dir,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Initial release of NariRaksha-100K dataset"
    )
    logger.info("Dataset published successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--repo_id", type=str, required=True, help="e.g., username/NariRaksha-100K")
    parser.add_argument("--token", type=str, required=True, help="HF Hub token")
    args = parser.parse_args()
    
    publish_dataset(args.dataset_dir, args.repo_id, args.token)
