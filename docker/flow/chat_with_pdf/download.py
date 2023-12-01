import requests
import os
import re

from utils.lock import acquire_lock
from utils.logging import log
from constants import PDF_DIR

def download(url: str) -> str:
    path = os.path.join(PDF_DIR, normalize_filename(url) + ".pdf")
    lock_path = path + ".lock"

    with acquire_lock(lock_path):
        if os.path.exists(path):
            log("PDF already exists in " + os.path.abspath(path))
            return path
        
        log("Downloading PDF from " + url)
        response = requests.get(url)

        with open(path, "wb") as f:
            f.write(response.content)

        return path
    
def normalize_filename(filename):
    return re.sub(r"[^\w\-_. ]", "_", filename)