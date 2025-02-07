import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

class CustomAugmentation:
    def __init__(self):
        self.affine_params = {
            'degrees': 10,      # Rotasi acak antara -10 dan 10 derajat
            'translate': (0.1, 0.1),  # Translasi acak hingga 10% dari ukuran gambar
        }

    def add_noise(self, image):
        image_np = np.array(image)
        mean = 0
        std = random.uniform(1, 10)  # Random standard deviation for noise
        noise = np.random.normal(mean, std, image_np.shape)
        noisy_image = image_np + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image), std

    def add_blur(self, image):
        blur_sigma = random.uniform(0.1, 2.0)  # Random sigma value for Gaussian blur
        return transforms.functional.gaussian_blur(image, kernel_size=(5, 5), sigma=blur_sigma), blur_sigma

    def add_contrast(self, image):
        contrast_value = random.uniform(0.5, 1.5)  # Random contrast value between 0.5 and 1.5
        image = transforms.functional.adjust_contrast(image, contrast_value)
        return image, contrast_value

    def __call__(self, image):
        applied_transforms = []

        # Randomly choose one of the transformations with affine
        transform_type = random.choice(['contrast', 'blur', 'noise'])

        if transform_type == 'contrast':
            image, contrast_value = self.add_contrast(image)
            applied_transforms.append(f"Contrast_{contrast_value:.2f}")

        elif transform_type == 'blur':
            image, blur_sigma = self.add_blur(image)
            applied_transforms.append(f"Blur_{blur_sigma:.2f}")

        elif transform_type == 'noise':
            image, noise_std = self.add_noise(image)
            applied_transforms.append(f"Noise_{noise_std:.2f}")

        # Apply random affine transformation
        affine_transform = transforms.RandomAffine(
            degrees=self.affine_params['degrees'],
            translate=self.affine_params['translate'],
        )
        image = affine_transform(image)
        applied_transforms.append("Affine")

        return image, applied_transforms

def augment_image(image_path, output_dir, augment_transform, base_name=None, iteration=0):
    # Load the image using PIL
    image = Image.open(image_path)
    if base_name is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Apply augmentations
    augmented_image, applied_transforms = augment_transform(image)

    # Build the augmentation suffix based on applied transforms
    augment_suffix = "_".join(applied_transforms)

    # Save augmented image with a unique name including augmentation info
    augmented_image.save(os.path.join(output_dir, f"{base_name}_augmented_{augment_suffix}_{iteration}.png"))

def augment_train_data(train_dir, output_dir, augment_transform, max_images_per_class=1600):
    for class_folder in tqdm(os.listdir(train_dir), desc="Processing classes"):
        class_folder_path = os.path.join(train_dir, class_folder)
        output_class_folder_path = os.path.join(output_dir, class_folder)

        if os.path.isdir(class_folder_path):
            os.makedirs(output_class_folder_path, exist_ok=True)

            # List all original images in the input directory
            original_images = [f for f in os.listdir(class_folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

            # Count the existing images in the output directory
            existing_images = [f for f in os.listdir(output_class_folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            current_image_count = len(existing_images)

            # Perform augmentation until we reach max_images_per_class
            iteration = 0
            while current_image_count < max_images_per_class:
                filename = random.choice(original_images)
                image_path = os.path.join(class_folder_path, filename)
                base_name = os.path.splitext(filename)[0]
                augment_image(image_path, output_class_folder_path, augment_transform, base_name=base_name, iteration=iteration)
                iteration += 1
                current_image_count += 1

                # Ensure we don't exceed the limit
                if current_image_count >= max_images_per_class:
                    break

    print("Augmentation complete!")

# Define augmentation transformations using CustomAugmentation
augment_transform = CustomAugmentation()

# Contoh penggunaan
# train_dir = '../../../.venv/dataset/dataset_fix_final/Train'  # Ganti dengan path ke data train asli Anda
# train_dir = '../../../.venv/dataset/dataset_nosegmentation_new/Train'  # Ganti dengan path ke data train asli Anda
# train_dir = '../../../.venv/dataset/dataset_cgrabcut_final/Train'  # Ganti dengan path ke data train asli Anda
train_dir = '../../../.venv/dataset/dataset_grabcut_final/Train'  # Ganti dengan path ke data train asli Anda
output_dir = '../../../.venv/dataset/dataset_grabcut_final/Train_augmented'  # Ganti dengan path ke folder output untuk hasil augmentasi
augment_train_data(train_dir, output_dir, augment_transform, max_images_per_class=1600)
