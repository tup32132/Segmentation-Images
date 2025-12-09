# pip install Office365-REST-Python-Client pydicom matplotlib

import io
import matplotlib.pyplot as plt
import pydicom
from office365.sharepoint.client_context import ClientContext
from office365.runtime.auth.user_credential import UserCredential

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Your SharePoint Site URL (e.g., https://contoso.sharepoint.com/sites/Radiology)
SHAREPOINT_URL = 'https://<YOUR_ORG>.sharepoint.com/sites/<YOUR_SITE>'

# 2. Your Credentials
# often works if the device/network is trusted.
USERNAME = 'your.email@company.com'
PASSWORD = 'your_password'

# 3. Path to the DICOM file on SharePoint
# Format: /sites/<site_name>/<library_name>/<folder>/<filename>.dcm
# Example: /sites/Radiology/Shared Documents/Scans/test_image.dcm
FILE_RELATIVE_URL = '/sites/<YOUR_SITE>/Shared Documents/<PATH_TO_FILE>.dcm'

# ==========================================
# MAIN SCRIPT
# ==========================================

def get_sharepoint_context(url, username, password):
    """Establishes connection to SharePoint."""
    try:
        user_credentials = UserCredential(username, password)
        ctx = ClientContext(url).with_credentials(user_credentials)
        
        # specific call to verify connection
        web = ctx.web
        ctx.load(web)
        ctx.execute_query()
        print(f"Connected to SharePoint site: {web.properties['Title']}")
        return ctx
    except Exception as e:
        print(f"Failed to authenticate. Error: {e}")
        return None

def download_dicom_in_memory(ctx, file_url):
    """Downloads file into a memory buffer (no local file saved)."""
    try:
        response = io.BytesIO()
        ctx.web.get_file_by_server_relative_url(file_url).download(response).execute_query()
        response.seek(0) # Reset pointer to start of file
        print(f"DICOM file downloaded into memory.")
        return response
    except Exception as e:
        print(f"Failed to download file. Check the FILE_RELATIVE_URL.")
        print(f"Error: {e}")
        return None

def display_dicom(file_stream):
    """Reads DICOM from memory and displays it."""
    try:
        # Read DICOM from memory
        ds = pydicom.dcmread(file_stream)
        
        # info dump
        print(f"--- DICOM INFO ---")
        print(f"Patient ID: {ds.get('PatientID', 'Unknown')}")
        print(f"Modality:   {ds.get('Modality', 'Unknown')}")
        
        # Display
        if 'PixelData' in ds:
            plt.figure(figsize=(6, 6))
            plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
            plt.title("SharePoint DICOM Test")
            plt.axis('off')
            plt.show()
            print("Image displayed.")
        else:
            print("DICOM read successfully, but contains no pixel data to display.")
            
    except Exception as e:
        print(f"Failed to parse/display DICOM. Error: {e}")

if __name__ == "__main__":
    print("--- Starting SharePoint Access Test ---")
    
    # 1. Connect
    ctx = get_sharepoint_context(SHAREPOINT_URL, USERNAME, PASSWORD)
    
    if ctx:
        # 2. Download
        dicom_stream = download_dicom_in_memory(ctx, FILE_RELATIVE_URL)
        
        if dicom_stream:
            # 3. Display
            display_dicom(dicom_stream)