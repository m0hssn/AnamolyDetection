import os
import requests
import tarfile

def download_and_extract_dataset(url, dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)
    os.chdir(dataset_dir)

    response = requests.get(url, stream=True)
    tar_file = url.split("/")[-1]

    with open(tar_file, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    with tarfile.open(tar_file, "r:gz") as tar:
        tar.extractall()

    os.remove(tar_file)
    print("Dataset downloaded and extracted successfully.")