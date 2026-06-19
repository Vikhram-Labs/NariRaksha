import os
import argparse
from huggingface_hub import HfApi, create_repo
from loguru import logger

def publish_model(model_dir: str, repo_id: str, token: str):
    logger.info(f"Publishing model from {model_dir} to {repo_id}")
    api = HfApi(token=token)
    
    # Create repo if not exists
    create_repo(repo_id, repo_type="model", token=token, exist_ok=True)
    
    # Upload files
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Initial release of NariRaksha SLM"
    )
    logger.info("Model published successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--repo_id", type=str, required=True, help="e.g., username/NariRaksha-Qwen-3B")
    parser.add_argument("--token", type=str, required=True, help="HF Hub token")
    args = parser.parse_args()
    
    publish_model(args.model_dir, args.repo_id, args.token)
