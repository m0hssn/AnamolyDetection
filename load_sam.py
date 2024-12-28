import requests

def download_file(url, save_path):
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Open the file in write-binary mode and save the content
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded and saved as {save_path}")
    else:
        print(f"Failed to download file. HTTP Status Code: {response.status_code}")

