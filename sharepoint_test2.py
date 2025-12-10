# pip install Office365-REST-Python-Client pydicom matplotlib
# import sys, subprocess; subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'Office365-REST-Python-Client'])
# import sys, subprocess; subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pydicom'])
# import sys, subprocess; subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib'])

import io
import matplotlib.pyplot as plt
import pydicom
# import json
from office365.sharepoint.client_context import ClientContext
from office365.runtime.auth.user_credential import UserCredential
# from office365.sharepoint.request import SharePointRequest
# from office365.sharepoint.files.file import File

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Your SharePoint Site URL 
SHAREPOINT_URL = 'https://uhhospitals.sharepoint.com/sites/Radiology2'

# 2. Your Credentials
# often works if the device/network is trusted.
USERNAME = 'kristy.hollingshead@UHhospitals.org'
PASSWORD = '<password>'

# 3. Your Cookies
RTFA = 'rtfa here'
FEDAUTH = 'fed auth here'

# 4. Path to the DICOM file on SharePoint
# Format: /sites/<site_name>/<library_name>/<folder>/<filename>.dcm
FILE_RELATIVE_URL = '/sites/Radiology2/Shared%20Documents/Sdrive/Rad%20RSI%20Berlin/Final_DICOMS_KMHtest/EE00BE5C.dcm'

# ==========================================
# MAIN SCRIPT
# ==========================================

def get_sharepoint_context_v2():
    """Establishes connection to SharePoint."""
    try:
        user_credentials = UserCredential(USERNAME, PASSWORD)
        ctx = ClientContext(SHAREPOINT_URL).with_credentials(user_credentials)
        
        # specific call to verify connection
        web = ctx.web
        ctx.load(web)
        ctx.execute_query()
        print(f"Connected to SharePoint site: {web.properties['Title']}")
        return ctx
    except Exception as e:
        print(f"Failed to authenticate. Error: {e}")
        return None

def get_sharepoint_context_v3():
    ctx = ClientContext(SHAREPOINT_URL)
    print(ctx)
    
    # PASTE YOUR COOKIES HERE
    # These strings will be very long
    ctx.authentication_context.set_cookie('rtFa', RTFA)
    ctx.authentication_context.set_cookie('FedAuth', FEDAUTH)    
    try:
        web = ctx.web
        ctx.load(web)
        ctx.execute_query()
        print(f"Connected using Browser Cookies!")
        return ctx
    except Exception as e:
        print(f"Cookie Auth Failed: {e}")
        return None

def enable_browser_cookies(ctx):
    """
    Registers a hook to inject authentication cookies into every request
    made by this ClientContext.
    """
    def _add_cookie_header(request):
        request.headers['Cookie'] += f"rtFa={RTFA}; FedAuth={FEDAUTH}"

    # Hook into the 'before_execute' event of the pending request
    
    ctx.pending_request().beforeExecute += _add_cookie_header

def get_sharepoint_context_v4():
    """Creates context and attaches cookie auth."""
    try:
        ctx = ClientContext(SHAREPOINT_URL)
        enable_browser_cookies(ctx, RTFA, FEDAUTH)
        print("enabled cookies?")
        
        # Verify connection by loading the web title
        web = ctx.web
        ctx.load(web)
        print("url",web.resource_url)
        # ctx.execute_query()
        print(f"Connected to SharePoint site with cookies.")
        return ctx
    except Exception as e:
        print(f"Failed to authenticate with cookies. Error: {e}")
        return None

def get_sharepoint_context():
    """Creates context and attaches cookie auth."""
    cookies = {
        "FedAuth": RTFA,
        "rtFA": FEDAUTH,
    }
    try:
        ctx = ClientContext(SHAREPOINT_URL).with_cookies(cookies)
        print("added cookie context?")

        # Verify connection
        web = ctx.web
        ctx.load(web)
        ctx.execute_query()
        print("url", web.properties['URL'])
        return ctx
    except Exception as e:
        print(f"Failed to authenticate with cookies. Error: {e}")
        return None

def download_dicom_in_memory(ctx):
    """Downloads file into a memory buffer (no local file saved)."""
    try:
        response = io.BytesIO()
        with open('sample.dcm', "wb") as local_file:
            f = ctx.web.get_file_by_server_relative_url(FILE_RELATIVE_URL)
            f.download(local_file)
            ctx.execute_query()
        print(f"DICOM file downloaded into memory")
        #response.seek(0) # Reset pointer to start of file
        return 'sample.dcm'
    except Exception as e:
        print(f"Failed to download file. Check the FILE_RELATIVE_URL.")
        print(f"Error: {e}")
        return None

def display_dicom(file_stream):
    """Reads DICOM from memory and displays it."""
    print("file:",file_stream)
    try:
        # Read DICOM from memory
        ds = pydicom.dcmread(file_stream, force=True)
        
        # info dump
        print(f"--- DICOM INFO ---")
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
    # ctx = get_sharepoint_context(SHAREPOINT_URL, USERNAME, PASSWORD)
    ctx = get_sharepoint_context()
    
    if ctx:
        # 2. Download
        dicom_stream = download_dicom_in_memory(ctx)
        
        if dicom_stream:
            # 3. Display
            display_dicom(dicom_stream)
