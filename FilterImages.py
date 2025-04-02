# import os
# import cv2
# import torch
# import numpy as np
# from ultralytics import YOLO  # Ensure you have the YOLOv5 model installed

# # Path to dataset folder
# dataset_path = "datasets/pedestrian"
# filtered_train_path = "filtered_data/train_RL_FLIR_Data"
# filtered_test_path = "filtered_data/test_RL_FLIR_Data"

# # Load YOLO model
# model = YOLO("yolov5su.pt")  # Using YOLOv5 small model

# # Create directories for filtered dataset
# os.makedirs(filtered_train_path, exist_ok=True)
# os.makedirs(filtered_test_path, exist_ok=True)

# # Define filtering criteria
# min_height = 120
# train_ratio = 0.7  # 70% for training, 30% for testing

# # Get all image paths
# image_paths = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path) if img.endswith(('.jpeg', '.png'))]

# # Print length of image list
# print(f"Total images found: {len(image_paths)}")

# # Check if images exist
# if len(image_paths) == 0:
#     print("No images found in dataset path. Please check the folder path or image formats.")
#     exit()

# # Shuffle for random split
# np.random.shuffle(image_paths)

# # Process each image
# train_count = 0
# test_count = 0

# for img_path in image_paths:
#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"Warning: Could not read image {img_path}")
#         continue
    
#     results = model(img)  # Run YOLO on the image
    
#     save_image = False
#     for result in results:
#         for box in result.boxes:
#             if int(box.cls) == 0:  # Class 0 corresponds to "person" in COCO dataset
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 height = y2 - y1
                
#                 if height >= min_height:
#                     save_image = True
#                     break
    
#     if save_image:
#         if train_count / len(image_paths) < train_ratio:
#             cv2.imwrite(os.path.join(filtered_train_path, os.path.basename(img_path)), img)
#             train_count += 1
#         else:
#             cv2.imwrite(os.path.join(filtered_test_path, os.path.basename(img_path)), img)
#             test_count += 1

# print(f"Filtered dataset: {train_count} training images, {test_count} testing images.")
import os
import cv2
import torch
import numpy as np
import pandas
from ultralytics import YOLO

# Paths
dataset_path = "datasets/pedestrian"
filtered_train_path = "filtered_data/train_RL_FLIR_Data"
filtered_test_path = "filtered_data/test_RL_FLIR_Data"

# Load YOLOv5
# model = YOLO("yolov5su.pt")  # Ensure the model is correctly downloaded
model = YOLO("yolov5su.pt")

# Create output directories
os.makedirs(filtered_train_path, exist_ok=True)
os.makedirs(filtered_test_path, exist_ok=True)

# Filtering criteria
min_height = 120
train_ratio = 0.7

# Get image paths
image_paths = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path) if img.endswith(('.jpeg', '.png'))]
np.random.shuffle(image_paths)

# Split data into train/test
split_idx = int(train_ratio * len(image_paths))
train_images, test_images = image_paths[:split_idx], image_paths[split_idx:]

def process_images(image_list, save_dir):
    count = 0
    for img_path in image_list:
        img = cv2.imread(img_path)
        results = model(img)  # Run YOLOv5 detection

        save_image = False
        for result in results:
            for box in result.boxes:
                cls = int(box.cls.item())  # Extract class index properly
                if cls == 0:  # Person class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    height = y2 - y1
                    if height >= min_height:
                        save_image = True
                        break
        
        if save_image:
            cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path)), img)
            count += 1
    return count

# Process and count images
train_count = process_images(train_images, filtered_train_path)
test_count = process_images(test_images, filtered_test_path)

print(f"Filtered dataset: {train_count} training images, {test_count} testing images.")
