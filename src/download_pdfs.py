# src/download_pdfs.py
# Verifies all 6 PDFs exist in data/pdfs/
# On HuggingFace Spaces, PDFs are uploaded directly to the repo.
# Downloads only if a file is missing as a fallback.

import os
import urllib.request

PDFS = {
    "code_on_wages_2019.pdf": "https://labour.gov.in/sites/default/files/TheCodeonWages2019.pdf",
    "industrial_relations_code_2020.pdf": "https://labour.gov.in/sites/default/files/TheIndustrialRelationsCode2020.pdf",
    "code_on_social_security_2020.pdf": "https://labour.gov.in/sites/default/files/SS-Code2020.pdf",
    "osh_code_2020.pdf": "https://labour.gov.in/sites/default/files/TheOccupationalSafetyHealthandWorkingConditionsCode2020.pdf",
    "epf_act_1952.pdf": "https://www.epfindia.gov.in/site_docs/PDFs/Downloads_PDFs/EPFAct1952.pdf",
    "posh_act_2013.pdf": "https://indiacode.nic.in/bitstream/123456789/15340/1/the_sexual_harassment_of_women_at_workplace_act%2C_2013.pdf",
}

PDF_FOLDER = os.path.join("data", "pdfs")


def download_pdfs():
    """
    Checks if all 6 PDFs exist on disk.
    If a PDF already exists and is valid, skips it.
    Only attempts download for missing files.
    Returns True if all files are present after checking.
    """
    os.makedirs(PDF_FOLDER, exist_ok=True)

    all_present = True

    for filename, url in PDFS.items():
        filepath = os.path.join(PDF_FOLDER, filename)

        # If file exists and is larger than 1KB it is valid — skip download
        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
            print(f"   ✓ Found: {filename} ({os.path.getsize(filepath)//1024} KB)")
            continue

        # File missing — try to download it
        print(f"   Downloading missing file: {filename}...")
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                with open(filepath, "wb") as f:
                    f.write(response.read())
            print(f"      ✓ Downloaded {filename}")
        except Exception as e:
            print(f"      ✗ Could not download {filename}: {e}")
            all_present = False

    return all_present