import argparse
import os
import re
import tkinter as tk
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Union

import cv2
import numpy as np
import pytesseract
import bisect
from PIL import Image, ImageTk
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from tqdm import tqdm

# Configure Tesseract and Poppler paths
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_BIN_DIR = r"C:\Users\david\miniconda3\envs\quiz-ingestion\Library\bin"

courses = connection.account.get_courses()
for course in courses["instructor"]:
    print(course)
for course in courses["student"]:
    print(course)


def select_roi_on_image(pil_image: Image.Image) -> Tuple[int, int, int, int]:
    """
    Displays a PIL Image in a Tkinter window for the user to select a bounding box (ROI).
    
    Returns:
        (x, y, w, h): A 4-element tuple representing the bounding box.
    """
    root = tk.Tk()
    root.title("Select ROI - Drag to select area, press ENTER to confirm, ESC to cancel")
    root.geometry("1920x1080")

    frame = tk.Frame(root, width=1920, height=1080)
    frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(frame, bd=0, highlightthickness=0)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    hbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
    hbar.pack(side=tk.BOTTOM, fill=tk.X)
    vbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
    vbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

    photo = ImageTk.PhotoImage(pil_image)
    img_width, img_height = pil_image.size
    canvas.config(scrollregion=(0, 0, img_width, img_height))
    canvas.create_image(0, 0, anchor='nw', image=photo)

    rect = None
    start_x, start_y = 0, 0
    roi = [0, 0, 0, 0]

    def on_button_press(event: tk.Event) -> None:
        nonlocal start_x, start_y, rect
        start_x = canvas.canvasx(event.x)
        start_y = canvas.canvasy(event.y)
        if rect:
            canvas.delete(rect)
        rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red')

    def on_move_press(event: tk.Event) -> None:
        nonlocal rect
        cur_x = canvas.canvasx(event.x)
        cur_y = canvas.canvasy(event.y)
        canvas.coords(rect, start_x, start_y, cur_x, cur_y)

    def on_key_press(event: tk.Event) -> None:
        if event.keysym == 'Return':
            if rect:
                x1, y1, x2, y2 = canvas.coords(rect)
                roi[0] = int(x1)
                roi[1] = int(y1)
                roi[2] = int(x2 - x1)
                roi[3] = int(y2 - y1)
            root.quit()
        elif event.keysym == 'Escape':
            roi[0] = roi[1] = roi[2] = roi[3] = 0
            root.quit()

    canvas.bind("<ButtonPress-1>", on_button_press)
    canvas.bind("<B1-Motion>", on_move_press)
    root.bind("<Key>", on_key_press)

    root.mainloop()
    root.destroy()

    return tuple(roi)  # (x, y, w, h)


def extract_number(roi_img: Image.Image) -> Optional[int]:
    """
    Perform OCR on the given ROI image to extract an integer page number.
    Returns None if no digits are found.
    """
    ocr_text = pytesseract.image_to_string(
        roi_img,
        config="--psm 7 -c tessedit_char_whitelist=0123456789"
    )
    match = re.search(r'(\d+)', ocr_text.strip())
    return int(match.group(1)) if match else None


def ocr_page_number_from_roi(pil_page: Image.Image, bbox: Tuple[int, int, int, int]) -> Optional[int]:
    """
    Given a PIL Image and bounding box, extracts a page number via OCR.
    """
    x, y, w, h = bbox
    page_np = cv2.cvtColor(np.array(pil_page), cv2.COLOR_RGB2BGR)
    roi = page_np[y : y + h, x : x + w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    roi_pil = Image.fromarray(thresh)

    return extract_number(roi_pil)


def convert_first_page_for_roi(pdf_path: str) -> Image.Image:
    """
    Converts only the first page of the PDF to a PIL image for ROI selection.
    """
    first_page = convert_from_path(
        pdf_path, 
        poppler_path=POPPLER_BIN_DIR, 
        first_page=1, 
        last_page=1
    )[0]
    return first_page


def convert_pdf_to_images(pdf_path: str) -> List[Image.Image]:
    """
    Converts all pages of a PDF to a list of PIL images.
    """
    pdf_reader = PdfReader(pdf_path)
    total_pages = len(pdf_reader.pages)

    pages = []
    for page_number in tqdm(range(1, total_pages + 1), desc="Converting PDF to images"):
        page = convert_from_path(
            pdf_path,
            poppler_path=POPPLER_BIN_DIR,
            first_page=page_number,
            last_page=page_number
        )[0]
        pages.append(page)
    return pages


def perform_ocr_on_pages(
    pages: List[Image.Image], 
    bbox: Tuple[int, int, int, int]
) -> List[Tuple[Optional[int], Image.Image]]:
    """
    Perform OCR on a list of page images, returning a list of tuples:
    (page_index, recognized_page_number, page_image).
    """
    recognized_pages = []
    for idx, page_img in tqdm(enumerate(pages, start=1), total=len(pages), desc="Performing OCR"):
        page_number = ocr_page_number_from_roi(page_img, bbox)
        recognized_pages.append((page_number, page_img))
    return recognized_pages


def analyze_page_numbers(
    recognized_pages: List[Tuple[Optional[int], Image.Image]],
    total_expected: int,
    per_test_count: int,
    actual_page_count: int
) -> Tuple[
    List[
        Union[
            Tuple[int, Dict[str, Union[List[int], bool]], List[Image.Image], Tuple[float, float], List[Optional[int]]],
            Tuple[int, str]
        ]
    ],
    List[
        Tuple[int, Dict[str, Union[List[int], bool]], List[Image.Image], Tuple[float, float], List[Optional[int]]]
    ]
]:
    """
    Analyzes recognized page numbers for anomalies.

    Returns:
        (anomalies_list, valid_list)

        anomalies_list:
            A list containing either:
              - (chunk_index, anomaly_dict, chunk_images, percentage_range, recognized_page_nums)
              - (chunk_index, "string describing overall anomaly")  # e.g. page count mismatch

        valid_list:
            A list of 5-tuples for valid chunks:
              - (chunk_index, anomaly_dict, chunk_images, percentage_range, recognized_page_nums)
    """
    
    def chunk_pages(recognized_pages: List[Tuple[Optional[int], Image.Image]]) \
            -> List[Tuple[Tuple[Optional[int], ...], List[Image.Image], Tuple[float, float]]]:
        """
        Breaks the recognized pages into chunks based on where the recognized page number = 1,
        and returns a list of tuples: (page_numbers_in_chunk, images_in_chunk, (start%, end%)).
        """
        sections = []
        current_section_numbers = []
        current_section_images = []
        total_pages = len(recognized_pages)

        for i, recognized_page in enumerate(recognized_pages):
            recognized_page_number, page_image = recognized_page

            # If we're at the very beginning and recognized_page_number != 1,
            # we treat it as part of a chunk.
            if i == 0 and recognized_page_number != 1:
                current_section_numbers.append(recognized_page_number)
                current_section_images.append(page_image)
                continue

            # If we see a "1" and we already have a chunk in progress,
            # close off that chunk and start a new one
            if recognized_page_number == 1 and current_section_numbers:
                start_percent = (i - len(current_section_numbers)) / total_pages * 100
                end_percent = i / total_pages * 100

                sections.append(
                    (tuple(current_section_numbers), current_section_images[:], (start_percent, end_percent))
                )

                # Start a new chunk
                current_section_numbers = []
                current_section_images = []

            # Whether it is a new chunk or continuing the old one, add this page
            current_section_numbers.append(recognized_page_number)
            current_section_images.append(page_image)

        # After looping, if there's a final chunk not yet appended:
        if current_section_numbers:
            start_percent = (total_pages - len(current_section_numbers)) / total_pages * 100
            end_percent = 100.0
            sections.append((tuple(current_section_numbers), current_section_images[:], (start_percent, end_percent)))

        return sections
    
    def detect_anomalies(sequence: List[Optional[int]], expected_pages: int) -> Dict[str, Union[List[int], bool]]:
        """
        Given a list of recognized page numbers (with None filtered out), returns a dictionary
        describing potential anomalies such as missing_pages, duplicate_pages, out_of_order, etc.
        """
        anomalies = {
            "missing_pages": [],
            "duplicate_pages": [],
            "likely_two_tests": False,
            "out_of_order": False,
            "has_none": False
        }

        # Check for missing pages
        expected_sequence = list(range(1, expected_pages + 1))
        anomalies["missing_pages"] = list(set(expected_sequence) - set(sequence))

        # Check for duplicate pages
        seen = set()
        duplicates = set()
        for page in sequence:
            if page in seen:
                duplicates.add(page)
            seen.add(page)
        anomalies["duplicate_pages"] = list(duplicates)

        # Check for out-of-order pages
        if sequence != sorted(sequence):
            anomalies["out_of_order"] = True
        
        # Check for likely two tests
        if len(sequence) > expected_pages and duplicates:
            anomalies["likely_two_tests"] = True
        
        if None in sequence:
            anomalies["has_none"] = True

        return anomalies

    def detect_anomalies_in_sections(
        pages: List[Tuple[Optional[int], Image.Image]],
        expected_pages: int
    ) -> Tuple[
        List[
            Union[
                Tuple[int, Dict[str, Union[List[int], bool]], List[Image.Image], Tuple[float, float], List[Optional[int]]],
                Tuple[int, str]
            ]
        ],
        List[Tuple[int, Dict[str, Union[List[int], bool]], List[Image.Image], Tuple[float, float], List[Optional[int]]]]
    ]:
        """
        Splits pages into chunks, detects anomalies in each chunk, and returns:
            (anomalies_list, valid_list)
        """
        sections = chunk_pages(pages)
        anomalies_list = []
        valid_list = []

        for i, (page_numbers, chunk_images, percentage) in enumerate(sections):
            recognized_page_nums = list(page_numbers)

            anomaly_dict = detect_anomalies(recognized_page_nums, expected_pages)

            # Check if this section has any anomalies
            if (
                anomaly_dict["missing_pages"]
                or anomaly_dict["duplicate_pages"]
                or anomaly_dict["out_of_order"]
                or anomaly_dict["likely_two_tests"]
                or anomaly_dict["has_none"]
            ):
                # 5-tuple: (chunk_index, anomaly_details, chunk_images, percentage_range, recognized_page_nums)
                anomalies_list.append((i, anomaly_dict, chunk_images, percentage, recognized_page_nums))
            else:
                # If no anomalies, put it in valid_list
                valid_list.append((i, anomaly_dict, chunk_images, percentage, recognized_page_nums))

        return anomalies_list, valid_list

    # Detect chunk-level anomalies vs. valid chunks
    anomalies_list, valid_list = detect_anomalies_in_sections(recognized_pages, per_test_count)

    # If the overall PDF page count doesn't match what we expect, that's an additional "global" anomaly
    if actual_page_count != total_expected:
        # This is a 2-tuple describing a general mismatch
        anomalies_list.append(
            (-1, f"Expected pages {total_expected} but received {actual_page_count}.")
        )

    return anomalies_list, valid_list

def save_anomalies_to_files(
    anomalies: List[
        Union[
            Tuple[int, Dict[str, Union[List[int], bool]], List[Image.Image], Tuple[float, float], List[Optional[int]]],
            Tuple[int, str]
        ]
    ],
    rectified_list: List[Tuple[int, List[Image.Image], List[int]]]
):
    """
    Saves each anomaly into a timestamped directory.
    If an anomaly's chunk_index is in rectified_list, writes 'Note: Automatically Rectified'.
    """
    if not anomalies:
        return

    # Create a base directory for this run's anomalies
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"anomaly_logs/{timestamp}"
    os.makedirs(base_dir, exist_ok=True)

    # Gather all chunk_indexes that were rectified
    rectified_indices = {chunk_index for (chunk_index, _, _) in rectified_list}

    for anomaly in anomalies:
        # --- CASE 1: Chunk-level anomaly with details (5 elements) ---
        # (chunk_index, anomaly_details, image_list, percentage_range, recognized_page_nums)
        if isinstance(anomaly, tuple) and len(anomaly) == 5:
            (chunk_index,
             anomaly_details,
             image_list,
             percentage_range,
             recognized_page_nums) = anomaly

            # Name the folder for the chunk
            if chunk_index == -1:
                chunk_dir = f"{base_dir}/general"
            else:
                chunk_dir = f"{base_dir}/Chunk_{chunk_index}"
            os.makedirs(chunk_dir, exist_ok=True)

            anomaly_file_path = f"{chunk_dir}/anomaly.txt"
            with open(anomaly_file_path, "w") as f:
                # Write basic info
                label = f"Chunk Index: {chunk_index if chunk_index != -1 else 'General'}"
                f.write(f"{label}\n")

                # Write location in PDF
                start_pct, end_pct = percentage_range
                if chunk_index != -1:
                    f.write(f"Location in stack: {start_pct:.2f}%-{end_pct:.2f}%\n")

                # Write details about anomalies
                f.write("Anomalies:\n")
                if anomaly_details.get("missing_pages"):
                    f.write(f"  Missing pages: {anomaly_details['missing_pages']}\n")
                if anomaly_details.get("duplicate_pages"):
                    f.write(f"  Duplicate pages: {anomaly_details['duplicate_pages']}\n")
                if anomaly_details.get("out_of_order"):
                    f.write("  Pages are out of order.\n")
                if anomaly_details.get("likely_two_tests"):
                    f.write("  This is likely two tests in one chunk.\n")
                if anomaly_details.get("has_none"):
                    f.write("  Some OCR attempts returned None.\n")

                # If the chunk is in rectified_list, add note
                if chunk_index in rectified_indices:
                    f.write("\nNote: Automatically Rectified\n")

            # Optionally save the first page’s image as a reference
            if image_list:
                first_image = image_list[0]
                if first_image:
                    first_image.save(os.path.join(chunk_dir, "image.png"))

        # --- CASE 2: General anomaly or mismatch (2 elements) ---
        elif isinstance(anomaly, tuple) and len(anomaly) == 2:
            chunk_index, description = anomaly

            if chunk_index == -1:
                chunk_dir = f"{base_dir}/general"
            else:
                chunk_dir = f"{base_dir}/Chunk_{chunk_index}"
            os.makedirs(chunk_dir, exist_ok=True)

            with open(f"{chunk_dir}/anomaly.txt", "w") as f:
                label = f"Chunk Index: {chunk_index if chunk_index != -1 else 'General'}"
                f.write(f"{label}\n")
                f.write(f"Description: {description}\n")

                if chunk_index in rectified_indices:
                    f.write("\nNote: Automatically Rectified\n")

def rectify_minor_missing_anomaly(
    anomalies: List[
        Union[
            Tuple[
                int, 
                Dict[str, Union[List[int], bool]], 
                List[Image.Image], 
                Tuple[float, float],
                List[Optional[int]]
            ],
            Tuple[int, str]
        ]
    ],
    k: int
) -> Tuple[
    List[
        Union[
            Tuple[int, Dict[str, Union[List[int], bool]], List[Image.Image], Tuple[float, float], List[Optional[int]]],
            Tuple[int, str]
        ]
    ],
    List[Tuple[int, List[Image.Image], List[int]]]
]:
    """
    Takes the anomalies list and attempts to 'rectify' any chunk whose anomalies are only:
      - Possibly out-of-order pages,
      - Missing pages (<= k),
      - No duplicates,
      - Not flagged as likely_two_tests.

    For each such chunk:
      1) Sort the chunk_images by their recognized page numbers (if out_of_order == True).
      2) Insert blank pages *in the correct spot* for each missing page (<= k).
         (We use bisect to place them between the closest recognized page numbers.)
      3) Remove that chunk from the anomalies and add it to 'rectified_list'.

    Args:
        anomalies: A list of anomaly entries, each of which might be:
          - (chunk_index,
             anomaly_details,
             chunk_images,
             percentage_range,
             recognized_page_nums)
          - (chunk_index, "string describing overall anomaly")
          
        k: The maximum number of missing pages allowed to auto-rectify.

    Returns:
        (new_anomalies_list, rectified_list):
            - new_anomalies_list:
                The same anomaly list but excluding the ones we successfully rectified here.
            - rectified_list:
                A list of tuples of the form (chunk_index, rectified_images, missing_pages),
                representing the chunks we successfully rectified.
                'rectified_images' is the newly sorted chunk images (if out_of_order)
                plus newly added blank pages inserted at the correct positions.
    """

    def create_blank_page(reference_image: Image.Image) -> Image.Image:
        """
        Create a blank page the same size as the reference_image.
        """
        width, height = reference_image.size
        blank = Image.new("RGB", (width, height), color=(255, 255, 255))
        return blank

    new_anomalies_list = []
    rectified_list = []

    for anomaly in anomalies:
        # We only handle chunk-level anomalies that are 5-tuples:
        #   (chunk_index, anomaly_details, chunk_images, (start_pct, end_pct), recognized_page_nums)
        if isinstance(anomaly, tuple) and len(anomaly) == 5:
            (chunk_index,
             anomaly_details,
             chunk_images,
             _percentage_range,
             recognized_page_nums) = anomaly

            missing_pages = anomaly_details.get("missing_pages", [])
            duplicate_pages = anomaly_details.get("duplicate_pages", [])
            out_of_order = anomaly_details.get("out_of_order", False)
            likely_two_tests = anomaly_details.get("likely_two_tests", False)
            has_none = anomaly_details.get("has_none", False)

            # Check conditions for auto-rectification
            if (
                len(missing_pages) <= k
                and not duplicate_pages
                and not likely_two_tests
                and not has_none
                and 1 not in missing_pages
            ):
                # 1) Separate numeric pages from None pages, so we can sort & insert properly.
                numeric_pairs = []
                for img, pg_num in zip(chunk_images, recognized_page_nums):
                    numeric_pairs.append((pg_num, img))
                
                # If out_of_order == True, sort numeric_pairs by page number
                if out_of_order:
                    numeric_pairs.sort(key=lambda x: x[0])  # x is (pg_num, image)

                # 2) Insert blank pages for each missing page in the correct spot
                sorted_missing = sorted(missing_pages)
                if numeric_pairs:
                    reference_img = numeric_pairs[0][1]
                else:
                    # fallback if numeric_pairs is empty
                    reference_img = Image.new("RGB", (612, 792), color=(255, 255, 255))

                recognized_only = [p[0] for p in numeric_pairs]
                for mp in sorted_missing:
                    blank_page = create_blank_page(reference_img)
                    # find where mp should be inserted in recognized_only
                    insert_index = bisect.bisect_left(recognized_only, mp)
                    # Insert into numeric_pairs
                    numeric_pairs.insert(insert_index, (mp, blank_page))
                    recognized_only.insert(insert_index, mp)

                # 3) Reconstruct final rectified_images & recognized_page_nums
                rectified_images = []
                rectified_numbers = []
                for pg_num, img in numeric_pairs:
                    rectified_images.append(img)
                    rectified_numbers.append(pg_num)

                # 4) Add to rectified_list
                rectified_list.append(
                    (chunk_index, rectified_images, missing_pages)
                )
            else:
                new_anomalies_list.append(anomaly)

        else:
            new_anomalies_list.append(anomaly)

    return new_anomalies_list, rectified_list

def save_combined_pdf(
    rectified_list: List[Tuple[int, List[Image.Image], List[int]]],
    valid_list: List[Tuple[int, Dict[str, Union[List[int], bool]], List[Image.Image], Tuple[float, float], List[Optional[int]]]],
    new_anomalies_list: Optional[List[
        Union[
            Tuple[int, Dict[str, Union[List[int], bool]], List[Image.Image], Tuple[float, float], List[Optional[int]]],
            Tuple[int, str]
        ]
    ]] = None,
    output_pdf_path: str = "combined_output.pdf"
) -> None:
    """
    Merge rectified chunks, valid chunks, and (optionally) remaining anomaly chunks
    into a single PDF, preserving the original chunk order.

    1. Pull out (chunk_index, [images]) from each list.
    2. Combine them into one list and sort by chunk_index ascending.
    3. Flatten all images in sorted order into a single list.
    4. Save the combined list of images as a PDF.

    :param rectified_list:
        List of (chunk_index, rectified_images, missing_pages).

    :param valid_list:
        List of 5-tuples 
          (chunk_index, anomaly_dict, images, percentage_range, recognized_page_nums).

    :param new_anomalies_list:
        Either None, or a list of entries:
          (chunk_index, anomaly_dict, images, percentage_range, recognized_page_nums)
          OR
          (chunk_index, "some string describing an anomaly").
        Only the 5-tuple entries contain images.

    :param output_pdf_path:
        Where to save the final merged PDF.
    """
    # 1) Build a combined list of (chunk_index, [images]) from each data source.
    combined_chunks: List[Tuple[int, List[Image.Image]]] = []

    # -- valid_list => (chunk_index, images)
    for (chunk_idx, _anomaly_dict, images, _pct, _recognized_nums) in valid_list:
        combined_chunks.append((chunk_idx, images))

    # -- rectified_list => (chunk_index, images)
    for (chunk_idx, images, _missing_pages) in rectified_list:
        combined_chunks.append((chunk_idx, images))

    # -- new_anomalies_list => only 5-tuple anomalies have images (and chunk_index != -1)
    if new_anomalies_list is not None:
        for anomaly in new_anomalies_list:
            if isinstance(anomaly, tuple) and len(anomaly) == 5:
                chunk_idx, _anom_dict, images, _pct, _recognized_nums = anomaly
                if chunk_idx != -1:
                    combined_chunks.append((chunk_idx, images))
    else:
        print("new_anomalies_list is None; no anomaly chunks will be included.")

    # 2) Sort all of them by chunk_index (ascending) so we output in original-like order
    combined_chunks.sort(key=lambda x: x[0])

    # 3) Flatten the images in sorted order
    ordered_images: List[Image.Image] = []
    for chunk_idx, images in combined_chunks:
        ordered_images.extend(images)

    if not ordered_images:
        print("No images found across all chunks; nothing to save.")
        return

    # Convert to 'RGB' mode so PIL can save them in a single PDF
    ordered_images_rgb = []
    for im in ordered_images:
        if im.mode != 'RGB':
            ordered_images_rgb.append(im.convert('RGB'))
        else:
            ordered_images_rgb.append(im)

    # 4) Save them all as a single PDF
    first_page = ordered_images_rgb[0]
    if len(ordered_images_rgb) == 1:
        # Only one page to save
        first_page.save(output_pdf_path, "PDF")
    else:
        # Multiple pages
        first_page.save(
            output_pdf_path,
            "PDF",
            save_all=True,
            append_images=ordered_images_rgb[1:]
        )

    print(f"Combined PDF saved to: {output_pdf_path}")


def main_process(pdf_path: str, total_expected: int, per_test_count: int) -> None:
    """
    Main workflow combining the smaller steps:
      1. Convert the first page for ROI selection and get the bounding box.
      2. Convert all pages to images.
      3. Perform OCR for each page.
      4. Analyze the recognized page numbers for anomalies.
      5. Save/report any anomalies found.
    """
    print("Preparing to convert the first page for ROI selection...")
    first_page = convert_first_page_for_roi(pdf_path)

    print("Displaying the first page for ROI selection...")
    bbox = select_roi_on_image(first_page)
    print(f"Selected bounding box: {bbox}")

    pages = convert_pdf_to_images(pdf_path)
    actual_page_count = len(pages)

    recognized_pages = perform_ocr_on_pages(pages, bbox)

    anomalies_list, valid_list = analyze_page_numbers(
        recognized_pages=recognized_pages,
        total_expected=total_expected,
        per_test_count=per_test_count,
        actual_page_count=actual_page_count
    )

    new_anomalies_list, rectified_list = rectify_minor_missing_anomaly(anomalies_list, k=3)

    save_anomalies_to_files(anomalies_list, rectified_list)

    save_combined_pdf(
        rectified_list=rectified_list,
        valid_list=valid_list,
        # new_anomalies_list=new_anomalies_list,
        output_pdf_path="combined_output.pdf"
    )

    # Print recognized results
    print(recognized_pages)
    print("\nRecognized page numbers in order:")
    for i, recognized_pages in enumerate(recognized_pages):
        print(f"  PDF Page {i}: recognized number = {recognized_pages[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract page numbers from scanned PDF.")
    parser.add_argument("pdf_path", type=str, help="Path to the scanned PDF file.")
    parser.add_argument("total_pages", type=int, help="Total expected page count in the PDF.")
    parser.add_argument("per_test_page_count", type=int, help="Number of pages each test should have.")
    args = parser.parse_args()

    main_process(
        pdf_path=args.pdf_path,
        total_expected=args.total_pages,
        per_test_count=args.per_test_page_count
    )

if __name__ == "__main__":
    main()
