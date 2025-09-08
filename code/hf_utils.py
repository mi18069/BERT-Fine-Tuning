import os
import json
import torch
import shutil
from transformers import AutoConfig
from huggingface_hub import login, HfApi, snapshot_download, upload_folder
from dotenv import load_dotenv

# For saving models, it's necessary to create write token on HF platform and log in with that credentials.
# This way only users with this token can publish models to HF. 
# Also create and pass read token which can be shared in order to get models from HF.

load_dotenv()

SAVE_DIR = "./tmp_model"
ACCOUNT_NAME = "Jovan23"

# Load tokens 
HF_READ_TOKEN = os.getenv("HF_READ_TOKEN")
HF_WRITE_TOKEN = os.getenv("HF_WRITE_TOKEN")
login(HF_WRITE_TOKEN)


def save_model_to_hf(model, repo_name, private=True, token=HF_WRITE_TOKEN):
    os.makedirs(SAVE_DIR, exist_ok=True)
    # Save config.json
    config = AutoConfig.from_pretrained("distilbert-base-uncased", num_labels=2)
    config.save_pretrained(SAVE_DIR)

    # Save model weights
    torch.save(model.state_dict(), f"{SAVE_DIR}/pytorch_model.bin")

    # Write minimal config.json
    with open(f"{SAVE_DIR}/config.json", "w") as f:
        f.write('{"model_type": "bert", "num_labels": 2}')

    # Create repo if doesn't exist and upload
    api = HfApi()
    api.create_repo(repo_id=f"{ACCOUNT_NAME}/{repo_name}", repo_type="model", private=private, exist_ok=True, token=token)

    upload_folder(
        repo_id=f"{ACCOUNT_NAME}/{repo_name}",
        repo_type="model",
        folder_path=SAVE_DIR,
        path_in_repo=".",
        token=token,
    )

    # Removing localy saved files
    shutil.rmtree(SAVE_DIR)
    print(f"Model pushed to https://huggingface.co/{ACCOUNT_NAME}/{repo_name}")


def load_model_from_hf(repo_name, model, token=HF_WRITE_TOKEN, device="cpu"):
    cache_dir = snapshot_download(repo_id=f"{ACCOUNT_NAME}/{repo_name}", repo_type="model", token=token)

    with open(f"{cache_dir}/config.json", "r") as f:
        config = json.load(f)

    state_dict = torch.load(f"{cache_dir}/pytorch_model.bin", map_location=device)
    model.load_state_dict(state_dict)

    print(f"Model loaded from https://huggingface.co/{ACCOUNT_NAME}/{repo_name}")
    return model

