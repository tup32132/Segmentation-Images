import browser_cookie3
import requests
import pydicom
import io
import matplotlib.pyplot as plt
from urllib.parse import quote

# --- CONFIGURATION ---
# The full URL to your SharePoint site
SHAREPOINT_URL = "https://yourcompany.sharepoint.com/sites/YourSiteName"

# The server-relative path to the file (e.g., /sites/YourSiteName/Shared Documents/...)
FILE_RELATIVE_PATH = "/sites/YourSiteName/Shared Documents/Folder/image.dcm"

def load_dicom_auto_auth():
    print("🔍 Attempting to grab cookies from your local browser...")
    
    try:
        # 1. Automatically load cookies from Edge (or try .chrome() / .firefox())
        # This looks into your browser's local storage for the SharePoint session
        cj = browser_cookie3.edge(domain_name='sharepoint.com')
        
        print("Cookies found. Attempting SharePoint connection...")

        # 2. Construct the URL for the file stream
        # We encode the path to handle spaces safely
        encoded_path = quote(FILE_RELATIVE_PATH)
        api_url = f"{SHAREPOINT_URL}/_api/web/GetFileByServerRelativeUrl('{encoded_path}')/$value"
        
        # 3. Request the file using the grabbed cookies
        response = requests.get(api_url, cookies=cj)
        
        if response.status_code == 200:
            print("Connection Successful! File downloaded.")
            return response.content
        elif response.status_code == 403:
            print("Access Denied (403).") 
            print("Tip: Open the specific SharePoint folder in your browser first to refresh the session.")
        elif response.status_code == 404:
            print(f"File not found (404) at: {api_url}")
        else:
            print(f"Error {response.status_code}: {response.reason}")

    except Exception as e:
        print(f"Error: {e}")
        print("Note: You must be logged into SharePoint in your browser for this to work.")
    
    return None

def display_dicom(file_bytes):
    if not file_bytes: return

    try:
        dicom_stream = io.BytesIO(file_bytes)
        ds = pydicom.dcmread(dicom_stream)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(ds.pixel_array)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error parsing DICOM: {e}")

if __name__ == "__main__":
    data = load_dicom_auto_auth()
    display_dicom(data)