import os
import cv2
import numpy as np
from tqdm import tqdm

# Path folder input dan output
input_folder = '../../../.venv/dataset/dataset_cgrabcut_original_new'
output_folder = '../../../.venv/dataset/dataset_cgrabcut_original_new_resized'
# input_folder = '../../../.venv/dataset/dataset_grabcut_ori'
# output_folder = '../../../.venv/dataset/dataset_grabcut_ori_resized'
target_size = (224, 224)

# Membuat folder output jika belum ada
os.makedirs(output_folder, exist_ok=True)

# Melakukan iterasi pada setiap subfolder di dalam folder input
for subdir in os.listdir(input_folder):
    subdir_path = os.path.join(input_folder, subdir)
    if os.path.isdir(subdir_path):
        # Membuat subfolder di dalam folder output
        output_subdir = os.path.join(output_folder, subdir)
        os.makedirs(output_subdir, exist_ok=True)

        # Mengambil daftar file di subfolder
        file_list = os.listdir(subdir_path)

        # Menggunakan tqdm untuk menampilkan progress bar
        for file_name in tqdm(file_list, desc=f"Processing {subdir}", unit="file"):
            file_path = os.path.join(subdir_path, file_name)
            if os.path.isfile(file_path):
                # Membaca gambar dengan cv2
                img = cv2.imread(file_path)

                # Pastikan gambar berhasil terbaca
                if img is None:
                    print(f"Failed to load image {file_path}")
                    continue

                # Resize gambar
                img_resized = cv2.resize(img, target_size)

                # Debug ukuran gambar setelah resize
                print(f"Resized {file_name}: {img_resized.shape}")

                # Menyimpan gambar ke folder output
                output_path = os.path.join(output_subdir, file_name)
                cv2.imwrite(output_path, img_resized)

print("Proses resize selesai.")
