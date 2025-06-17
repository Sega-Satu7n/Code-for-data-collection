import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os
import json

# ASL alphabet and settings for data collection
ASL_ALPHABET = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','space','nothing'
]

# Mapping from letter to folder name (same for English)
FOLDER_MAPPING = {letter: letter for letter in ASL_ALPHABET}

# Reverse mapping (needed if loading from previous runs)
REVERSE_MAPPING = {v: k for k, v in FOLDER_MAPPING.items()}

# Parameters
IMAGES_PER_CLASS = 10  # desired number of samples per letter
BASE_FOLDER = "asl_train_data"
CONFIG_FILE = "asl_progress.json"

class DatasetCollector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=1, detectionCon=0.8)
        self.offset = 20
        self.imgSize = 224
        self.min_hand_size = 50
        self.capture_delay = 0.5
        self.last_capture_time = 0
        self.imgWhite = None

        # Create base directory
        os.makedirs(BASE_FOLDER, exist_ok=True)
        # Create per-letter folders
        for letter in ASL_ALPHABET:
            os.makedirs(os.path.join(BASE_FOLDER, FOLDER_MAPPING[letter]), exist_ok=True)

        # Load or initialize progress
        self.progress = self.load_progress()
        print("\nCurrent progress:")
        for letter, count in self.progress.items():
            if count > 0:
                print(f"Letter {letter}: {count}/{IMAGES_PER_CLASS}")

        # Determine next letter
        self.current_letter = self.get_next_letter()
        self.counter = self.progress.get(self.current_letter, 0) if self.current_letter else 0

        if self.current_letter:
            print(f"\nStarting data collection for letter: {self.current_letter}")
            print("Controls:")
            print("S - save image")
            print("N - skip current letter")
            print("R - repeat last save")
            print("Q - quit")
        else:
            print("\nAll letters are collected!")

    def load_progress(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    converted = {}
                    for key, value in data.items():
                        # support both letter keys and folder keys
                        letter = REVERSE_MAPPING.get(key, key)
                        converted[letter] = value
                    return converted
            except Exception as e:
                print(f"Error loading progress: {e}")
        # default zero counts
        return {letter: 0 for letter in ASL_ALPHABET}

    def save_progress(self):
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.progress, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving progress: {e}")

    def get_next_letter(self):
        for letter in ASL_ALPHABET:
            if self.progress.get(letter, 0) < IMAGES_PER_CLASS:
                return letter
        return None

    def preprocess_hand_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        img = cv2.merge((cl, a, b))
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def resize_and_pad(self, img_crop, size):
        h, w = img_crop.shape[:2]
        aspect = h / w
        if aspect > 1:
            new_h = size
            new_w = int(size / aspect)
        else:
            new_w = size
            new_h = int(size * aspect)
        resized = cv2.resize(img_crop, (new_w, new_h))
        canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
        x = (size - new_w) // 2 if aspect > 1 else 0
        y = (size - new_h) // 2 if aspect <= 1 else 0
        canvas[y:y+new_h, x:x+new_w] = resized
        return canvas

    def create_info_display(self, img):
        info_h = 120
        info = np.ones((info_h, img.shape[1], 3), dtype=np.uint8) * 255
        cv2.putText(info, f"Letter: {self.current_letter}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
        cv2.putText(info, f"Progress: {self.counter}/{IMAGES_PER_CLASS}", (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
        percent = sum(self.progress.values()) / (len(ASL_ALPHABET)*IMAGES_PER_CLASS) * 100
        cv2.putText(info, f"Overall: {percent:.1f}%", (10, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
        return np.vstack([info, img])

    def save_image(self):
        if self.imgWhite is None:
            print("No image to save")
            return False
        folder = os.path.join(BASE_FOLDER, FOLDER_MAPPING[self.current_letter])
        filename = os.path.join(folder, f"Image_{time.time():.3f}.jpg")
        if cv2.imwrite(filename, self.imgWhite):
            self.counter += 1
            self.progress[self.current_letter] = self.counter
            self.save_progress()
            print(f"Saved {self.counter}/{IMAGES_PER_CLASS} for {self.current_letter}")
            if self.counter >= IMAGES_PER_CLASS:
                prev = self.current_letter
                self.current_letter = self.get_next_letter()
                self.counter = self.progress.get(self.current_letter, 0) if self.current_letter else 0
                print(f"Finished {prev}, next: {self.current_letter}")
            return True
        print("Error saving image")
        return False

    def skip_letter(self):
        if not self.current_letter:
            return False
        prev = self.current_letter
        self.current_letter = self.get_next_letter()
        self.counter = self.progress.get(self.current_letter, 0)
        print(f"Skipped {prev}, next: {self.current_letter}")
        return True

    def run(self):
        last_img = None
        while True:
            if not self.current_letter:
                print("All data collected!")
                break
            success, frame = self.cap.read()
            if not success:
                break
            hands, img = self.detector.findHands(frame)
            if hands:
                x,y,w,h = hands[0]['bbox']
                if w>=self.min_hand_size and h>=self.min_hand_size:
                    crop = frame[y-self.offset:y+h+self.offset, x-self.offset:x+w+self.offset]
                    if crop.size:
                        white = self.resize_and_pad(crop, self.imgSize)
                        self.imgWhite = self.preprocess_hand_image(white)
                        cv2.imshow("Crop", crop)
                        cv2.imshow("Processed", self.imgWhite)
                        cv2.rectangle(img, (x-self.offset,y-self.offset),(x+w+self.offset,y+h+self.offset),(255,0,255),4)
                else:
                    cv2.putText(img, "Move hand closer", (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), 2)
            display = self.create_info_display(img)
            cv2.imshow("ASL Data Collector", display)
            key = cv2.waitKey(1)
            now = time.time()
            if key==ord('s') and now-self.last_capture_time>=self.capture_delay:
                if self.save_image():
                    self.last_capture_time = now
                    last_img = self.imgWhite.copy()
            elif key==ord('r') and last_img is not None:
                self.imgWhite = last_img.copy()
                if self.save_image(): print("Re-saved last image")
            elif key==ord('n'):
                if not self.skip_letter(): break
            elif key==ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nFinal stats:")
        for letter, cnt in self.progress.items():
            if cnt>0:
                print(f"{letter}: {cnt}/{IMAGES_PER_CLASS}")

if __name__ == "__main__":
    collector = DatasetCollector()
    collector.run()
