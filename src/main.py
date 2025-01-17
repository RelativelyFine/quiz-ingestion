import os
import cv2
import numpy as np
import pytesseract
import argparse
from pdf2image import convert_from_path
from PIL import Image
import re
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
poppler_bin_dir = r"C:\Users\david\miniconda3\envs\quiz-ingestion\Library\bin"

def select_roi_on_image(pil_image):
    """
    Show a PIL Image in an OpenCV window so the user can drag a bounding box.
    Returns (x, y, w, h).
    """
    open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    bbox = cv2.selectROI(
        "Select ROI - Press ENTER or SPACE to confirm, 'c' to cancel",
        open_cv_image,
        fromCenter=False,
        showCrosshair=False
    )
    cv2.destroyAllWindows()

    return bbox  # (x, y, w, h)

def extract_number(roi_img):
    """
    Perform OCR on the grayscale thresholded ROI and return (if possible) only digits.
    We'll use a whitelist and page segmentation that favors a single line/block.
    Then we'll parse out digits with a regex to handle double-digit or multi-digit numbers.
    """
    text = pytesseract.image_to_string(
        roi_img,
        config="--psm 7 -c tessedit_char_whitelist=0123456789"
    )
    text = text.strip()
    match = re.search(r'(\d+)', text)
    if match:
        return int(match.group(1))
    return None

def ocr_page_number_from_roi(pil_page, bbox):
    """
    Given a PIL page image and a bounding box (x, y, w, h),
    crop the region, threshold it, and use Tesseract to read numeric text.
    Returns an integer page number or None if not recognized.
    """
    x, y, w, h = bbox
    page_np = cv2.cvtColor(np.array(pil_page), cv2.COLOR_RGB2BGR)
    roi = page_np[y:y+h, x:x+w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    roi_pil = Image.fromarray(thresh)

    number = extract_number(roi_pil)
    return number

def save_pil_image(pil_img, out_path):
    """
    Save a PIL image to out_path (e.g., 'reports/anomaly-2025-01-17/page1.jpg').
    """
    pil_img.save(out_path)

def main():
    parser = argparse.ArgumentParser(description="Extract page numbers from scanned PDF.")
    parser.add_argument("pdf_path", type=str, help="Path to the scanned PDF file.")
    parser.add_argument("total_pages", type=int, help="Total expected page count in the PDF.")
    parser.add_argument("per_test_page_count", type=int, help="Number of pages each test should have.")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    total_expected = args.total_pages
    per_test_count = args.per_test_page_count

    pages = convert_from_path(pdf_path, poppler_path=poppler_bin_dir)

    actual_page_count = len(pages)
    if actual_page_count != total_expected:
        print(f"Warning: The PDF has {actual_page_count} pages, but user input said {total_expected}.")

    print("Displaying the first page for ROI selection...")
    bbox = select_roi_on_image(pages[0])  # (x, y, w, h)
    print(f"Selected bounding box: {bbox}")

    recognized_pages = []
    for idx, page_img in enumerate(pages, start=1):
        page_number = ocr_page_number_from_roi(page_img, bbox)
        recognized_pages.append((idx, page_number, page_img))

    anomalies = []
    anomaly_images = []

    if actual_page_count != total_expected:
        anomalies.append(
            f"Mismatch page count: Expected {total_expected}, got {actual_page_count}."
        )

    # The logic: chunk the recognized pages in groups of per_test_page_count
    # Each chunk is expected to have pages [1, 2, 3, ..., per_test_page_count].
    # The "1st page" in each chunk is the reset point, i.e. page number = 1,
    # second page = 2, etc. We do a straightforward chunk approach: for i in range(0, total, per_test_count).
    # If the last chunk is incomplete, that’s also an anomaly.

    total_chunks = actual_page_count // per_test_count
    remainder = actual_page_count % per_test_count

    for chunk_idx in range(total_chunks):
        start = chunk_idx * per_test_count
        end = start + per_test_count
        chunk = recognized_pages[start:end]

        expected_seq = list(range(1, per_test_count + 1))
        actual_seq = [p[1] for p in chunk]

        mismatches = []
        for i in range(per_test_count):
            if actual_seq[i] != expected_seq[i]:
                mismatches.append(
                    f"Page in PDF index={chunk[i][0]}: recognized_number={actual_seq[i]} != expected={expected_seq[i]}"
                )

        if mismatches:

            anomalies.append(f"Chunk #{chunk_idx+1} has page-order issues:")
            anomalies.extend(mismatches)

            first_page_img = chunk[0][2]
            global_page_index = chunk[0][0]
            anomaly_images.append((
                first_page_img,
                f"chunk{chunk_idx+1}_page{global_page_index}.jpg"
            ))

    if remainder > 0:
        anomalies.append(f"PDF ends with a partial chunk of {remainder} pages; expected {per_test_count}.")

        leftover_start = total_chunks * per_test_count
        leftover_chunk = recognized_pages[leftover_start:]
        if leftover_chunk:
            first_page_img = leftover_chunk[0][2]
            global_page_index = leftover_chunk[0][0]
            anomaly_images.append((
                first_page_img,
                f"leftover_chunk_page{global_page_index}.jpg"
            ))

    if anomalies:
        anomaly_date = datetime.now().strftime("%Y-%m-%d")
        anomaly_dir = os.path.join("reports", f"anomaly-{anomaly_date}")
        os.makedirs(anomaly_dir, exist_ok=True)

        issues_path = os.path.join(anomaly_dir, "issues.txt")
        with open(issues_path, "w", encoding="utf-8") as f:
            for line in anomalies:
                f.write(line + "\n")

        for (img, filename) in anomaly_images:
            out_path = os.path.join(anomaly_dir, filename)
            save_pil_image(img, out_path)

        print("[!] Anomalies found. See:", issues_path)
    else:
        print("[+] No anomalies detected.")

    print("\nRecognized page numbers in order:")
    for idx, number, _ in recognized_pages:
        print(f"  PDF-Page {idx}: recognized number = {number}")

if __name__ == "__main__":
    main()
