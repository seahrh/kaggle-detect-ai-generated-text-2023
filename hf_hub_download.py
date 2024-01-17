from huggingface_hub import hf_hub_download

# see https://huggingface.co/docs/huggingface_hub/quick-start

if __name__ == "__main__":
    repo_id = "meta-llama/Llama-2-13b-hf"
    filename = "pytorch_model-00003-of-00003.bin"
    print(repo_id)
    hf_hub_download(repo_id=repo_id, filename=filename)
