import os
from huggingface_hub import hf_hub_download  # Load model directly
# filename：文件名称
hf_hub_download(repo_id="internlm/internlm-7b", filename="config.json", local_dir='./')