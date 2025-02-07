import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pickle

def apply_median_filter(mask, ksize=3):
    return cv2.medianBlur(mask, ksize)

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        return (xmin, ymin, xmax, ymax)

def apply_contour_detection(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask_contour = np.zeros_like(mask)
    cv2.drawContours(mask_contour, contours, -1, (255), thickness=cv2.FILLED)
    return mask_contour

def fill_holes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cv2.drawContours(mask, contours, i, 255, -1)
    return mask

def modified_mask(mask, rect, line_width=90, shrink_factor=0.9):
    xmin, ymin, xmax, ymax = rect
    width = xmax - xmin
    height = ymax - ymin

    # Bagian 1: Mask kotak
    if height > width:
        fg_xmin = int(xmin + (xmax - xmin) * 0.20)
        fg_ymin = int(ymin + (ymax - ymin) * 0.15)
        fg_xmax = int(xmax - (xmax - xmin) * 0.20)
        fg_ymax = int(ymax - (ymax - ymin) * 0.15)
    else:
        # Jika width lebih besar atau sama dengan height
        fg_xmin = int(xmin + (xmax - xmin) * 0.15)
        fg_ymin = int(ymin + (ymax - ymin) * 0.20)
        fg_xmax = int(xmax - (xmax - xmin) * 0.15)
        fg_ymax = int(ymax - (ymax - ymin) * 0.20)
    mask[fg_ymin:fg_ymax, fg_xmin:fg_xmax] = 1

    # Bagian 2: Garis "+" dengan shrink_factor
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2
    new_width = int(width * shrink_factor)
    new_height = int(height * shrink_factor)
    xmin_new = center_x - new_width // 2
    xmax_new = center_x + new_width // 2
    ymin_new = center_y - new_height // 2
    ymax_new = center_y + new_height // 2

    # Garis vertikal dari tanda "+"
    vert_ymin = ymin_new
    vert_ymax = ymax_new
    vert_xmin = max(center_x - line_width // 2, xmin_new)
    vert_xmax = min(center_x + line_width // 2, xmax_new)
    mask[vert_ymin:vert_ymax, vert_xmin:vert_xmax] = 1

    # Garis horizontal dari tanda "+"
    horiz_xmin = xmin_new
    horiz_xmax = xmax_new
    horiz_ymin = max(center_y - line_width // 2, ymin_new)
    horiz_ymax = min(center_y + line_width // 2, ymax_new)
    mask[horiz_ymin:horiz_ymax, horiz_xmin:horiz_xmax] = 1

    # Set the rest as probable background
    mask[mask == 0] = 2

    return mask

def process_image(image_path, xml_path, output_path):
    rect = parse_xml(xml_path)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")

    mask = np.zeros(img.shape[:2], np.uint8)

    # Create background and foreground models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Refine initial mask
    mask = modified_mask(mask, rect, line_width=90, shrink_factor=0.9)

    # Apply GrabCut with refined mask
    cv2.grabCut(img, mask, None, bgd_model, fgd_model, 10, cv2.GC_INIT_WITH_MASK)
    # Convert mask to binary
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') * 255

    # Restrict mask strictly within the bounding box
    xmin, ymin, xmax, ymax = rect
    strict_mask = np.zeros_like(mask)
    strict_mask[ymin:ymax, xmin:xmax] = mask[ymin:ymax, xmin:xmax]
    mask = strict_mask

    # Apply median filter to reduce noise
    mask_filtered = apply_median_filter(mask, ksize=3)

    # Apply contour detection and fill holes in the contour 
    mask_contour = apply_contour_detection(mask_filtered)
    mask_filled = fill_holes(mask_contour)

    # Perform bitwise AND to get the segmented image
    result = cv2.bitwise_and(img, img, mask=mask_filled)

    # Save result
    cv2.imwrite(output_path, result)

def process_folder(base_dir, output_base_dir, progress_file='progress.pkl'):
    if os.path.exists(progress_file):
        with open(progress_file, 'rb') as f:
            processed_images = pickle.load(f)
    else:
        processed_images = set()

    splits = ['Daun Bercak', 'Daun Karat', 'Daun Sehat']

    for split in splits:
        split_dir = os.path.join(base_dir, split)
        output_split_dir = os.path.join(output_base_dir, split)

        # Buat folder split jika belum ada
        os.makedirs(output_split_dir, exist_ok=True)

        for filename in tqdm(os.listdir(split_dir), desc=f'Processing {split}'):
            if filename.endswith(('.jpg', '.jpeg', '.png')):  # Sesuaikan dengan ekstensi gambar Anda
                image_path = os.path.join(split_dir, filename)
                xml_path = os.path.splitext(image_path)[0] + '.xml'
                output_path = os.path.join(output_split_dir, os.path.splitext(filename)[0] + '_segmented.png')

                if image_path in processed_images:
                    continue  # Skip already processed images

                if os.path.exists(xml_path):
                    try:
                        process_image(image_path, xml_path, output_path)
                        processed_images.add(image_path)
                    except Exception as e:
                        print(f"Failed to process {image_path}: {e}")

                    # Save progress after each image
                    with open(progress_file, 'wb') as f:
                        pickle.dump(processed_images, f)

    print("Processing complete!")

# Contoh penggunaan
base_dir = '../../../.venv/dataset/dataset_roboflow/'  # Path ke dataset asli Anda
output_base_dir = '../../../.venv/dataset/dataset_segmented_fix'
process_folder(base_dir, output_base_dir)
