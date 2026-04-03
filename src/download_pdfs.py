# src/download_pdfs.py
# Downloads all 6 official Indian government PDFs at startup.
# This runs automatically on HuggingFace Spaces since we cannot
# push binary PDF files to the repository.

import os
import urllib.request

# Official government PDF URLs
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
    Downloads all 6 PDFs from official government URLs.
    Skips files that already exist on disk.
    Returns True if all files are present, False if any download failed.
    """
    os.makedirs(PDF_FOLDER, exist_ok=True)

    all_success = True

    for filename, url in PDFS.items():
        filepath = os.path.join(PDF_FOLDER, filename)

        # Skip if already downloaded
        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
            print(f"   ✓ Already exists: {filename}")
            continue

        print(f"   Downloading: {filename}...")

        try:
            # Download with a browser-like user agent
            # Some government servers reject requests without a user agent
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                with open(filepath, "wb") as f:
                    f.write(response.read())

            size_kb = os.path.getsize(filepath) // 1024
            print(f"      ✓ Downloaded {filename} ({size_kb} KB)")

        except Exception as e:
            print(f"      ✗ Failed to download {filename}: {e}")
            all_success = False

    return all_success


if __name__ == "__main__":
    print("Downloading PDFs...")
    success = download_pdfs()
    if success:
        print("All PDFs downloaded successfully!")
    else:
        print("Some PDFs failed to download.")