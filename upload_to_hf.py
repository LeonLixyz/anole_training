from huggingface_hub import HfApi
import os
import argparse
import glob

def upload_model_to_hf(model_path, repo_id, ignore_patterns=["checkpoint-*"]):
    """Upload model files to Huggingface Hub, excluding files matching ignore patterns."""
    api = HfApi()
    
    # Create the repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True)
        print(f"Repository {repo_id} is ready.")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
    
    # Get all files in the model directory
    all_files = []
    for root, _, files in os.walk(model_path):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, model_path)
            all_files.append((file_path, rel_path))
    
    # Filter out files matching ignore patterns
    files_to_upload = []
    for file_path, rel_path in all_files:
        should_ignore = False
        for pattern in ignore_patterns:
            if any(glob.fnmatch.fnmatch(part, pattern) for part in rel_path.split(os.sep)):
                should_ignore = True
                break
        if not should_ignore:
            files_to_upload.append((file_path, rel_path))
    
    # Upload the files
    print(f"Uploading {len(files_to_upload)} files to {repo_id}...")
    for file_path, rel_path in files_to_upload:
        print(f"Uploading {rel_path}...")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=rel_path,
            repo_id=repo_id,
            commit_message=f"Upload {rel_path}"
        )
    
    print(f"Model uploaded successfully to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload model to Huggingface Hub")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--repo_id", type=str, required=True, help="Huggingface Hub repository ID (username/repo-name)")
    parser.add_argument("--ignore", type=str, nargs="+", default=["checkpoint-*"], help="Patterns to ignore")
    
    args = parser.parse_args()
    upload_model_to_hf(args.model_path, args.repo_id, args.ignore) 