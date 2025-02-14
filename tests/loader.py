import requests
import os

def download_file(url, save_path):
    """
    Downloads a file from a given URL and saves it to the specified path.
    
    :param url: The URL of the file to download.
    :param save_path: The path (including filename) where the file will be saved.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Write the file to the specified path
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"File downloaded successfully and saved to: {save_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")

# Example usage
if __name__ == "__main__":
    # URL of the file to download
    file_url = input("Enter the URL of the file to download: ")

    # Path where the file will be saved
    save_location = input("Enter the path to save the file (including filename): ")

    # Download the file
    download_file(file_url, save_location)