import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from PIL import Image

# ASL alphabet and settings (copied from dataColector.py)
ASL_ALPHABET = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'nothing'
]

# Parameters
BASE_FOLDER = "asl_train_data"
TARGET_IMAGES_PER_CLASS = 100
AUGMENTED_FOLDER = "asl_augmented_data"  # If you want to save to a different folder

class DataAugmenter:
    def __init__(self, source_folder=BASE_FOLDER, target_folder=None):
        self.source_folder = source_folder
        # If target_folder is None, augment in-place
        self.target_folder = target_folder if target_folder else source_folder
        
        # Create target directories if they don't exist
        if target_folder:
            os.makedirs(target_folder, exist_ok=True)
            for letter in ASL_ALPHABET:
                os.makedirs(os.path.join(target_folder, letter), exist_ok=True)
    
    def load_images(self, letter):
        """Load all images for a specific letter using PIL instead of cv2"""
        images = []
        img_folder = os.path.join(self.source_folder, letter)
        if not os.path.exists(img_folder):
            print(f"Warning: Folder {img_folder} does not exist")
            return images
        
        for filename in os.listdir(img_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(img_folder, filename)
                try:
                    # Use PIL to load the image and convert to numpy array
                    pil_img = Image.open(img_path)
                    # Convert PIL image to numpy array (RGB)
                    img = np.array(pil_img)
                    # Convert RGB to BGR (OpenCV format)
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        img = img[:, :, ::-1].copy()  # RGB to BGR
                    images.append(img)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        return images
    
    def save_augmented_image(self, img, letter, index):
        """Save an augmented image to the target folder using PIL"""
        target_dir = os.path.join(self.target_folder, letter)
        os.makedirs(target_dir, exist_ok=True)
        filename = f"Aug_{letter}_{index}.jpg"
        filepath = os.path.join(target_dir, filename)
        
        # Convert BGR to RGB (OpenCV to PIL)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = img[:, :, ::-1]  # BGR to RGB
        
        # Save using PIL
        pil_img = Image.fromarray(img)
        pil_img.save(filepath)
    
    def random_rotation(self, img, max_angle=15):
        """Apply random rotation to image"""
        angle = random.uniform(-max_angle, max_angle)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Create rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply affine transformation
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    def random_translation(self, img, max_shift=20):
        """Apply random translation to image"""
        h, w = img.shape[:2]
        tx = random.randint(-max_shift, max_shift)
        ty = random.randint(-max_shift, max_shift)
        
        # Create translation matrix
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        
        # Apply affine transformation
        translated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return translated
    
    def random_zoom(self, img, max_factor=0.15):
        """Apply random zoom to image"""
        factor = 1.0 + random.uniform(-max_factor, max_factor)
        h, w = img.shape[:2]
        h_new, w_new = int(h * factor), int(w * factor)
        
        # Compute crop coordinates
        x1 = max(0, (w_new - w) // 2)
        y1 = max(0, (h_new - h) // 2)
        x2 = min(w_new, x1 + w)
        y2 = min(h_new, y1 + h)
        
        # Resize and crop if needed
        if factor > 1.0:
            # Upscale
            resized = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
            result = resized[y1:y2, x1:x2]
            
            # Ensure the result is of the right size
            if result.shape[:2] != (h, w):
                result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            # Downscale
            # Create a blank canvas
            canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
            resized = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
            
            # Placement coordinates
            x_offset = (w - w_new) // 2
            y_offset = (h - h_new) // 2
            
            # Place the resized image on the canvas
            canvas[y_offset:y_offset+h_new, x_offset:x_offset+w_new] = resized
            result = canvas
            
        return result
    
    def random_brightness_contrast(self, img, max_brightness=30, max_contrast=0.3):
        """Apply random brightness and contrast adjustments"""
        # Brightness adjustment
        brightness = random.randint(-max_brightness, max_brightness)
        contrast = 1.0 + random.uniform(-max_contrast, max_contrast)
        
        # Apply brightness
        img = np.clip(img + brightness, 0, 255).astype(np.uint8)
        
        # Apply contrast
        img = np.clip(img * contrast, 0, 255).astype(np.uint8)
        
        return img
    
    def random_gaussian_blur(self, img, max_sigma=1.5):
        """Apply random Gaussian blur"""
        sigma = random.uniform(0.1, max_sigma)
        return cv2.GaussianBlur(img, (5, 5), sigma)
    
    def random_noise(self, img, noise_level=15):
        """Add random noise to image"""
        noise = np.random.normal(0, noise_level, img.shape).astype(np.int32)
        noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return noisy_img
    
    def random_perspective(self, img, distortion_scale=0.1):
        """Apply random perspective transformation"""
        h, w = img.shape[:2]
        
        # Get four corners
        points1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        
        # Random displacement of corners
        shift = int(w * distortion_scale)
        points2 = np.float32([
            [random.randint(0, shift), random.randint(0, shift)],
            [random.randint(w-shift, w), random.randint(0, shift)],
            [random.randint(0, shift), random.randint(h-shift, h)],
            [random.randint(w-shift, w), random.randint(h-shift, h)]
        ])
        
        # Get perspective transform matrix
        M = cv2.getPerspectiveTransform(points1, points2)
        
        # Apply perspective transformation
        warped = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return warped
    
    def apply_random_augmentations(self, img):
        """Apply multiple random augmentations to an image"""
        # Each augmentation has a chance to be applied
        augmented = img.copy()
        
        # Always apply at least one augmentation
        augmentations = [
            (self.random_rotation, 0.7),
            (self.random_translation, 0.7),
            (self.random_zoom, 0.5),
            (self.random_brightness_contrast, 0.7),
            (self.random_gaussian_blur, 0.3),
            (self.random_noise, 0.3),
            (self.random_perspective, 0.3)
        ]
        
        # Ensure at least one augmentation is applied
        if all(random.random() > prob for _, prob in augmentations):
            # If none were selected, choose one randomly
            aug_func, _ = random.choice(augmentations)
            augmented = aug_func(augmented)
        else:
            # Apply selected augmentations
            for aug_func, prob in augmentations:
                if random.random() < prob:
                    augmented = aug_func(augmented)
        
        return augmented
    
    def augment_letter(self, letter):
        """Augment images for a specific letter to reach the target count"""
        images = self.load_images(letter)
        
        if not images:
            print(f"No images found for letter {letter}")
            return
        
        current_count = len(images)
        print(f"Found {current_count} images for letter {letter}")
        
        # If we already have enough images, we're done
        if current_count >= TARGET_IMAGES_PER_CLASS:
            print(f"Already have {current_count} images for {letter}, skipping augmentation")
            return
        
        # Calculate how many more images we need
        needed = TARGET_IMAGES_PER_CLASS - current_count
        
        # Generate new images through augmentation
        for i in tqdm(range(needed), desc=f"Augmenting {letter}"):
            # Randomly select an image to augment
            base_img = random.choice(images)
            
            # Apply augmentations
            augmented = self.apply_random_augmentations(base_img)
            
            # Save the augmented image
            self.save_augmented_image(augmented, letter, current_count + i)
    
    def augment_all(self):
        """Augment all letters in the dataset"""
        for letter in ASL_ALPHABET:
            print(f"\nProcessing letter: {letter}")
            self.augment_letter(letter)
        
        print("\nAugmentation complete!")

if __name__ == "__main__":
    augmenter = DataAugmenter(source_folder=BASE_FOLDER, target_folder=BASE_FOLDER)
    augmenter.augment_all() 