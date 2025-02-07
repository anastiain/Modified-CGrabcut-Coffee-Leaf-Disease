import os
import hashlib
from collections import defaultdict

# Definisikan path ke folder utama dataset
dataset_base_path = '../../../.venv/dataset/dataset_roboflow_gabung/'

# Daftar subfolder
base_folders = ['Daun Bercak', 'Daun Karat', 'Daun Sehat']

# Fungsi untuk menghitung hash file
def hash_file(file_path):
    hash_func = hashlib.md5()  # Anda bisa menggunakan algoritma hashing lain seperti SHA-1
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)
    return hash_func.hexdigest()

# Fungsi untuk menghapus file duplikat
def remove_duplicates(base_folder):
    hashes = defaultdict(list)
    for root, _, files in os.walk(base_folder):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_hash = hash_file(file_path)
            hashes[file_hash].append(file_path)

    duplicates_removed = 0
    for file_list in hashes.values():
        if len(file_list) > 1:
            for file_path in file_list[1:]:  # Keep the first file, delete the rest
                os.remove(file_path)
                duplicates_removed += 1
                print(f"Removed duplicate: {file_path}")

    return duplicates_removed

# Hapus duplikat untuk setiap folder (training, valid, test)
total_duplicates_removed = 0
for base_folder in base_folders:
    folder_path = os.path.join(dataset_base_path, base_folder)
    total_duplicates_removed += remove_duplicates(folder_path)

print(f"Total {total_duplicates_removed} duplicate images removed.")
