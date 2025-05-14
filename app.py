from google.colab import drive
drive.mount('/content/drive')

from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Define save path in Google Drive
drive_save_path = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/Chinese_Landscape_Painting2photo"

# Ensure destination directory exists
os.makedirs(drive_save_path, exist_ok=True)

print(f"✅ Google Drive mounted and dataset will be saved at: {drive_save_path}")


import zipfile

# Define dataset ZIP path (Make sure this path is correct)
zip_path = "/content/drive/My Drive/Colab Notebooks/OGdata.zip"
extract_path = "/content/extracted_data"

# Extract ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("✅ Dataset extracted successfully!")

# Verify extracted folders
extracted_folders = os.listdir(extract_path)
print("📂 Extracted Folders:", extracted_folders)


import os
import shutil

# Define dataset root
dataset_root = "/content/extracted_data/Chinese_Landscape_Painting2photo"

# Define Google Drive save path
drive_save_path = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/Chinese_Landscape_Painting2photo"

# Ensure destination directory exists
os.makedirs(drive_save_path, exist_ok=True)

# Function to copy files one by one
def copy_files_individually(src_folder, dest_folder):
    if os.path.exists(src_folder):
        os.makedirs(dest_folder, exist_ok=True)  # Ensure target folder exists
        files = os.listdir(src_folder)
        total_files = len(files)

        for idx, file_name in enumerate(files, 1):
            src_file = os.path.join(src_folder, file_name)
            dest_file = os.path.join(dest_folder, file_name)

            if os.path.isfile(src_file):
                shutil.copy2(src_file, dest_file)  # Copy file with metadata

            if idx % 100 == 0:
                print(f"✅ Copied {idx}/{total_files} files in {os.path.basename(src_folder)}")

        print(f"✅ Completed: {os.path.basename(src_folder)} ({total_files} files copied)")
    else:
        print(f"❌ {os.path.basename(src_folder)} not found!")

# Copy each dataset folder
for folder in ["trainA", "trainB", "testA", "testB"]:
    copy_files_individually(os.path.join(dataset_root, folder), os.path.join(drive_save_path, folder))

# Verify copied files
for folder in ["trainA", "trainB", "testA", "testB"]:
    folder_path = os.path.join(drive_save_path, folder)
    if os.path.exists(folder_path):
        num_images = len(os.listdir(folder_path))
        print(f"📸 {folder}: {num_images} images in Google Drive")
    else:
        print(f"❌ {folder} is missing in Google Drive!")


import os

# Define the path to your dataset folders in Google Drive
dataset_path = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/Chinese_Landscape_Painting2photo"

# List of folders to sort
folders = ["trainA", "trainB", "testA", "testB"]

for folder in folders:
    folder_path = os.path.join(dataset_path, folder)
    if os.path.exists(folder_path):
        sorted_files = sorted(os.listdir(folder_path))  # Sort files alphabetically
        print(f"✅ Sorted {len(sorted_files)} images in {folder}")
        print(sorted_files[:10])  # Show first 10 sorted filenames for verification
    else:
        print(f"❌ Folder {folder} not found!")


import shutil

def sort_images(folder_path):
    # List all files and sort them
    files = sorted(os.listdir(folder_path))

    # Create a temporary folder for sorted images
    sorted_folder = folder_path + "_sorted"
    os.makedirs(sorted_folder, exist_ok=True)

    for index, file_name in enumerate(files, start=1):
        old_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(sorted_folder, file_name)

        if os.path.isfile(old_path):
            shutil.move(old_path, new_path)

    # Move sorted files back to original folder
    for file_name in sorted(os.listdir(sorted_folder)):
        shutil.move(os.path.join(sorted_folder, file_name), folder_path)

    os.rmdir(sorted_folder)  # Remove the temporary folder
    print(f"✅ Sorted: {folder_path}")

# Apply sorting to each dataset folder
for folder in ["trainA", "trainB", "testA", "testB"]:
    sort_images(os.path.join(dataset_path, folder))


import os

# Define original and new preprocessed folder paths
original_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/Chinese_Landscape_Painting2photo"
preprocessed_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/preprocessed_Chinese_Landscape_Painting2photo"

# Create preprocessed folder if it doesn't exist
os.makedirs(preprocessed_folder, exist_ok=True)

# Subfolders (trainA, trainB, testA, testB)
for subfolder in ["trainA", "trainB", "testA", "testB"]:
    os.makedirs(os.path.join(preprocessed_folder, subfolder), exist_ok=True)

print("✅ Preprocessed directories created successfully!")


import os
import cv2
import numpy as np
from tqdm import tqdm

# Define paths
original_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/Chinese_Landscape_Painting2photo"
preprocessed_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/preprocessed_Chinese_Landscape_Painting2photo"

# Create preprocessed folder if it doesn't exist
os.makedirs(preprocessed_folder, exist_ok=True)

# Subfolders (trainA, trainB, testA, testB)
for subfolder in ["trainA", "trainB", "testA", "testB"]:
    os.makedirs(os.path.join(preprocessed_folder, subfolder), exist_ok=True)

# Image Processing Parameters
IMG_SIZE = (256, 256)  # Resize dimensions

# Function to preprocess images
def preprocess_images(subfolder):
    input_path = os.path.join(original_folder, subfolder)
    output_path = os.path.join(preprocessed_folder, subfolder)

    print(f"🔄 Processing {subfolder}...")

    for img_name in tqdm(sorted(os.listdir(input_path))):
        img_path = os.path.join(input_path, img_name)
        save_path = os.path.join(output_path, img_name)

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            continue  # Skip corrupted images

        # Resize
        img = cv2.resize(img, IMG_SIZE)

        # Normalize (scale pixel values between 0 and 1)
        img = img / 255.0

        # Convert to uint8 before further processing
        img_uint8 = (img * 255).astype(np.uint8)

        # Edge Detection (Canny)
        edges = cv2.Canny(img_uint8, 100, 200)

        # Convert to Grayscale
        img_gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)

        # Histogram Equalization
        img_eq = cv2.equalizeHist(img_gray)

        # Save Preprocessed Images
        cv2.imwrite(save_path, img_uint8)  # Save normalized image
        cv2.imwrite(save_path.replace(".jpg", "_edges.jpg"), edges)  # Save edge-detected version
        cv2.imwrite(save_path.replace(".jpg", "_gray.jpg"), img_gray)  # Save grayscale version
        cv2.imwrite(save_path.replace(".jpg", "_hist_eq.jpg"), img_eq)  # Save histogram equalized version

    print(f"✅ {subfolder} Preprocessing Complete! Saved in {output_path}")

# Process all datasets
for subfolder in ["trainA", "trainB", "testA", "testB"]:
    preprocess_images(subfolder)

print("🚀 All preprocessing completed successfully!")


import os

base_path = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/preprocessed_Chinese_Landscape_Painting2photo"

for folder in ["trainA", "trainB", "testA", "testB"]:
    folder_path = os.path.join(base_path, folder)
    if os.path.exists(folder_path):
        print(f"📂 {folder}: {len(os.listdir(folder_path))} images")
    else:
        print(f"❌ {folder} does not exist!")


!ls "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/preprocessed_Chinese_Landscape_Painting2photo/testA"
!ls "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/preprocessed_Chinese_Landscape_Painting2photo/testB"


import cv2
import matplotlib.pyplot as plt
import os

# Define paths
base_path = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/preprocessed_Chinese_Landscape_Painting2photo"

# Select a folder to check
folders = ["trainA", "trainB", "testA", "testB"]

# Display one image from each folder
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    img_name = os.listdir(folder_path)[0]  # Pick the first image
    img_path = os.path.join(folder_path, img_name)

    img = cv2.imread(img_path)  # Read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    plt.figure()
    plt.imshow(img)
    plt.title(f"Sample from {folder}")
    plt.axis("off")

plt.show()


import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# Paths
preprocessed_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/preprocessed_Chinese_Landscape_Painting2photo"
feature_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/extracted_features_Chinese_Landscape_Painting2photo"

# Create feature storage folder
os.makedirs(feature_folder, exist_ok=True)

# Define subfolders
subfolders = ["trainA", "trainB", "testA", "testB"]

# Function to extract GLCM (Texture Features)
def extract_glcm_features(gray_img):
    glcm = graycomatrix(gray_img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return np.array([contrast, energy, homogeneity])

# Function to extract LBP (Texture Features)
def extract_lbp_features(gray_img):
    lbp = local_binary_pattern(gray_img, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)
    return hist

# Function to process images and extract features
def extract_features(subfolder):
    input_path = os.path.join(preprocessed_folder, subfolder)
    output_path = os.path.join(feature_folder, subfolder)
    os.makedirs(output_path, exist_ok=True)

    print(f"🔄 Extracting features from {subfolder}...")

    features = []

    for img_name in tqdm(sorted(os.listdir(input_path))):
        if "_edges.jpg" in img_name or "_hist_eq.jpg" in img_name:
            continue  # Skip non-original preprocessed images

        img_path = os.path.join(input_path, img_name)
        gray_path = img_path.replace(".jpg", "_gray.jpg")  # Load grayscale version

        # Load images
        img = cv2.imread(img_path)
        gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)

        if img is None or gray is None:
            continue  # Skip if image is not found

        # Extract Edge Features (Canny)
        edges = cv2.Canny(gray, 100, 200)
        edge_mean = np.mean(edges)
        edge_std = np.std(edges)

        # Extract Texture Features (GLCM & LBP)
        glcm_features = extract_glcm_features(gray)
        lbp_features = extract_lbp_features(gray)

        # Extract Color Features (Mean & Std Dev of RGB channels)
        mean_colors = np.mean(img, axis=(0, 1))  # [Mean_R, Mean_G, Mean_B]
        std_colors = np.std(img, axis=(0, 1))  # [Std_R, Std_G, Std_B]

        # Combine all features into one array
        feature_vector = np.concatenate([mean_colors, std_colors, glcm_features, lbp_features, [edge_mean, edge_std]])

        # Append to list
        features.append(feature_vector)

    # Save extracted features
    save_path = os.path.join(output_path, f"{subfolder}_features.npy")
    np.save(save_path, np.array(features))
    print(f"✅ {subfolder} Features Saved at: {save_path}")

# Run Feature Extraction for all subfolders
for subfolder in subfolders:
    extract_features(subfolder)

print("🚀 Feature Extraction Completed!")


import numpy as np

# Define feature paths
feature_paths = {
    "trainA": "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/extracted_features_Chinese_Landscape_Painting2photo/trainA/trainA_features.npy",
    "trainB": "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/extracted_features_Chinese_Landscape_Painting2photo/trainB/trainB_features.npy",
    "testA": "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/extracted_features_Chinese_Landscape_Painting2photo/testA/testA_features.npy",
    "testB": "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/extracted_features_Chinese_Landscape_Painting2photo/testB/testB_features.npy"
}

# Check feature files
for subset, path in feature_paths.items():
    try:
        features = np.load(path)
        print(f"✅ {subset} Feature Shape: {features.shape}")
        print(f"🔹 Sample Features ({subset}):\n", features[:3])  # Print first 3 feature vectors
    except FileNotFoundError:
        print(f"❌ {subset} feature file not found at: {path}")
    except Exception as e:
        print(f"⚠️ Error loading {subset}: {e}")


from google.colab import drive
drive.mount('/content/drive')


import numpy as np

# Define feature paths
feature_paths = {
    "trainA": "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/extracted_features_Chinese_Landscape_Painting2photo/trainA/trainA_features.npy",
    "trainB": "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/extracted_features_Chinese_Landscape_Painting2photo/trainB/trainB_features.npy",
    "testA": "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/extracted_features_Chinese_Landscape_Painting2photo/testA/testA_features.npy",
    "testB": "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/extracted_features_Chinese_Landscape_Painting2photo/testB/testB_features.npy"
}

# Check feature files
for subset, path in feature_paths.items():
    try:
        features = np.load(path)
        print(f"✅ {subset} Feature Shape: {features.shape}")
        print(f"🔹 Sample Features ({subset}):\n", features[:3])  # Print first 3 feature vectors
    except FileNotFoundError:
        print(f"❌ {subset} feature file not found at: {path}")
    except Exception as e:
        print(f"⚠️ Error loading {subset}: {e}")


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load extracted features
X_trainA = np.load(feature_paths["trainA"])
X_trainB = np.load(feature_paths["trainB"])
X_testA = np.load(feature_paths["testA"])
X_testB = np.load(feature_paths["testB"])

# Create Labels
y_trainA = np.zeros(X_trainA.shape[0])  # Label 0 for trainA
y_trainB = np.ones(X_trainB.shape[0])   # Label 1 for trainB
y_testA = np.zeros(X_testA.shape[0])    # Label 0 for testA
y_testB = np.ones(X_testB.shape[0])     # Label 1 for testB

# Combine training and testing data
X_train = np.vstack([X_trainA, X_trainB])
y_train = np.hstack([y_trainA, y_trainB])
X_test = np.vstack([X_testA, X_testB])
y_test = np.hstack([y_testA, y_testB])

# Split dataset for training and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(f"✅ Training Set: {X_train.shape}, Labels: {y_train.shape}")
print(f"✅ Validation Set: {X_val.shape}, Labels: {y_val.shape}")
print(f"✅ Testing Set: {X_test.shape}, Labels: {y_test.shape}")


# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Validate the model
y_pred_val = clf.predict(X_val)
val_acc = accuracy_score(y_val, y_pred_val)
print(f"✅ Validation Accuracy: {val_acc:.4f}")


# Test the model
y_pred_test = clf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"✅ Test Accuracy: {test_acc:.4f}")
print("🔹 Classification Report:\n", classification_report(y_test, y_pred_test))
print("🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))


from google.colab import drive
drive.mount('/content/drive')

import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define paths
base_path = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/preprocessed_Chinese_Landscape_Painting2photo"
output_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/fused_results"
os.makedirs(output_folder, exist_ok=True)

# Folders to fuse
folders = ["testA", "testB"]

# Get sorted filenames
image_namesA = sorted([f for f in os.listdir(os.path.join(base_path, "testA")) if f.endswith(".jpg") and "_gray" not in f and "_edges" not in f and "_hist_eq" not in f])
image_namesB = sorted([f for f in os.listdir(os.path.join(base_path, "testB")) if f.endswith(".jpg") and "_gray" not in f and "_edges" not in f and "_hist_eq" not in f])

min_len = min(len(image_namesA), len(image_namesB))

# Fuzzy Fusion Logic
for i in tqdm(range(min_len)):
    nameA = image_namesA[i]
    nameB = image_namesB[i]

    # Load original, grayscale, and edge images
    pathA = os.path.join(base_path, "testA", nameA)
    grayA = cv2.imread(pathA.replace(".jpg", "_gray.jpg"), cv2.IMREAD_GRAYSCALE)
    edgeA = cv2.Canny(grayA, 100, 200)

    pathB = os.path.join(base_path, "testB", nameB)
    grayB = cv2.imread(pathB.replace(".jpg", "_gray.jpg"), cv2.IMREAD_GRAYSCALE)
    edgeB = cv2.Canny(grayB, 100, 200)

    imgA = cv2.imread(pathA)
    imgB = cv2.imread(pathB)

    # Fuzzy-inspired logic (based on contrast and edge strength)
    contrastA = np.var(grayA)
    contrastB = np.var(grayB)
    edge_meanA = np.mean(edgeA)
    edge_meanB = np.mean(edgeB)

    # Fuzzy weights (adjustable)
    contrast_weight = contrastA / (contrastA + contrastB + 1e-5)
    edge_weight = edge_meanB / (edge_meanA + edge_meanB + 1e-5)
    trad_weight = 0.6 * contrast_weight + 0.4 * (1 - edge_weight)
    ai_weight = 1 - trad_weight

    # Fusion
    fused = cv2.addWeighted(imgA, trad_weight, imgB, ai_weight, 0)

    # Save
    fused_name = f"fused_{i+1:04d}.jpg"
    fused_path = os.path.join(output_folder, fused_name)
    cv2.imwrite(fused_path, fused)

    # Optional: Show one fused image in Colab
    if i == 0:
        plt.imshow(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB))
        plt.title("Sample Fused Output")
        plt.axis("off")
        plt.show()

print(f"✅ All fused images saved to: {output_folder}")


import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Paths
base_path = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/preprocessed_Chinese_Landscape_Painting2photo"
output_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/fused_results"
os.makedirs(output_folder, exist_ok=True)

# Load sorted image names
image_namesA = sorted([f for f in os.listdir(os.path.join(base_path, "testA")) if f.endswith(".jpg") and "_gray" not in f and "_edges" not in f and "_hist_eq" not in f])
image_namesB = sorted([f for f in os.listdir(os.path.join(base_path, "testB")) if f.endswith(".jpg") and "_gray" not in f and "_edges" not in f and "_hist_eq" not in f])

min_len = min(len(image_namesA), len(image_namesB))

# Fusion and visualization
for i in tqdm(range(min_len)):
    nameA = image_namesA[i]
    nameB = image_namesB[i]

    # Load grayscale and edge
    pathA = os.path.join(base_path, "testA", nameA)
    grayA = cv2.imread(pathA.replace(".jpg", "_gray.jpg"), cv2.IMREAD_GRAYSCALE)
    edgeA = cv2.Canny(grayA, 100, 200)

    pathB = os.path.join(base_path, "testB", nameB)
    grayB = cv2.imread(pathB.replace(".jpg", "_gray.jpg"), cv2.IMREAD_GRAYSCALE)
    edgeB = cv2.Canny(grayB, 100, 200)

    imgA = cv2.imread(pathA)
    imgB = cv2.imread(pathB)

    # Fuzzy weights
    contrastA = np.var(grayA)
    contrastB = np.var(grayB)
    edge_meanA = np.mean(edgeA)
    edge_meanB = np.mean(edgeB)

    contrast_weight = contrastA / (contrastA + contrastB + 1e-5)
    edge_weight = edge_meanB / (edge_meanA + edge_meanB + 1e-5)
    trad_weight = 0.6 * contrast_weight + 0.4 * (1 - edge_weight)
    ai_weight = 1 - trad_weight

    # Fusion
    fused = cv2.addWeighted(imgA, trad_weight, imgB, ai_weight, 0)

    # Save
    fused_name = f"fused_{i+1:04d}.jpg"
    fused_path = os.path.join(output_folder, fused_name)
    cv2.imwrite(fused_path, fused)

    # Display first few examples
    if i < 5:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB))
        plt.title("Input: Traditional (testA)")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB))
        plt.title("Input: AI Painting (testB)")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB))
        plt.title("Fused Output")
        plt.axis("off")
        plt.show()

print(f"✅ Fused images saved in: {output_folder}")


import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Paths
tcp_path = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/preprocessed_Chinese_Landscape_Painting2photo/testA"
ai_path = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/preprocessed_Chinese_Landscape_Painting2photo/testB"
fused_path = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/fused_output_TCP_preserved"
os.makedirs(fused_path, exist_ok=True)

# Get image filenames
tcp_images = sorted([f for f in os.listdir(tcp_path) if f.endswith(".jpg") and "_gray" not in f])
ai_images = sorted([f for f in os.listdir(ai_path) if f.endswith(".jpg") and "_gray" not in f])
total = min(len(tcp_images), len(ai_images))

# Fusion Process
for i in tqdm(range(total)):
    tcp_img_path = os.path.join(tcp_path, tcp_images[i])
    ai_img_path = os.path.join(ai_path, ai_images[i])

    # Read images
    tcp_img = cv2.imread(tcp_img_path)
    ai_img = cv2.imread(ai_img_path)

    # Resize to ensure match
    tcp_img = cv2.resize(tcp_img, (256, 256))
    ai_img = cv2.resize(ai_img, (256, 256))

    # Convert to grayscale
    grayA = cv2.cvtColor(tcp_img, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(ai_img, cv2.COLOR_BGR2GRAY)

    # Calculate simple contrast
    contrastA = np.var(grayA)
    contrastB = np.var(grayB)

    # Canny edge mean
    edgeA = cv2.Canny(grayA, 100, 200)
    edgeB = cv2.Canny(grayB, 100, 200)
    edge_meanA = np.mean(edgeA)
    edge_meanB = np.mean(edgeB)

    # Fuzzy-inspired fusion weights
    contrast_weight = contrastA / (contrastA + contrastB + 1e-5)
    edge_weight = edge_meanA / (edge_meanA + edge_meanB + 1e-5)

    # Favor TCP: assign higher weight to testA (tcp_img)
    tcp_weight = 0.7 * contrast_weight + 0.3 * edge_weight
    ai_weight = 1 - tcp_weight

    # Fuse images
    fused = cv2.addWeighted(tcp_img, tcp_weight, ai_img, ai_weight, 0)

    # Save
    fused_filename = f"fused_{i+1:04d}.jpg"
    cv2.imwrite(os.path.join(fused_path, fused_filename), fused)

    # Show every 100th image
    if i % 100 == 0:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(tcp_img, cv2.COLOR_BGR2RGB))
        plt.title("Traditional (testA)")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(ai_img, cv2.COLOR_BGR2RGB))
        plt.title("AI (testB)")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB))
        plt.title("Fused Output")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

print("✅ Fusion completed with TCP structure preserved.")


from google.colab import drive
drive.mount('/content/drive')

import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define folders
base_drive = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset"
tcp_folder = os.path.join(base_drive, "preprocessed_Chinese_Landscape_Painting2photo", "testA")  # Traditional
ai_folder = os.path.join(base_drive, "preprocessed_Chinese_Landscape_Painting2photo", "testB")   # AI
fused_folder = os.path.join(base_drive, "fused_outputs_tcp_preserved")
os.makedirs(fused_folder, exist_ok=True)

# List images
tcp_images = sorted([f for f in os.listdir(tcp_folder) if f.endswith(".jpg") and "_gray" not in f and "_edges" not in f and "_hist_eq" not in f])
ai_images = sorted([f for f in os.listdir(ai_folder) if f.endswith(".jpg") and "_gray" not in f and "_edges" not in f and "_hist_eq" not in f])

# Fuzzy fusion function
def fuse_images_tcp_preserved(imgA, imgB):
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    # Texture and edge-based fuzzy weights
    contrastA = np.var(grayA)
    contrastB = np.var(grayB)

    edgeA = cv2.Canny(grayA, 100, 200)
    edgeB = cv2.Canny(grayB, 100, 200)

    edge_meanA = np.mean(edgeA)
    edge_meanB = np.mean(edgeB)

    # Fuzzy weights (more weight to TCP structure)
    contrast_weight = contrastA / (contrastA + contrastB + 1e-5)
    edge_weight = edge_meanB / (edge_meanA + edge_meanB + 1e-5)

    tcp_weight = 0.7 * contrast_weight + 0.3 * (1 - edge_weight)
    ai_weight = 1 - tcp_weight

    fused = cv2.addWeighted(imgA, tcp_weight, imgB, ai_weight, 0)
    return fused

# Fuse and visualize a few examples
for i in range(5):  # Just show 5 examples
    img_nameA = tcp_images[i]
    img_nameB = ai_images[i]

    pathA = os.path.join(tcp_folder, img_nameA)
    pathB = os.path.join(ai_folder, img_nameB)

    imgA = cv2.imread(pathA)
    imgB = cv2.imread(pathB)

    if imgA is None or imgB is None:
        continue

    imgA = cv2.resize(imgA, (256, 256))
    imgB = cv2.resize(imgB, (256, 256))

    fused_img = fuse_images_tcp_preserved(imgA, imgB)

    # Save fused image
    save_path = os.path.join(fused_folder, f"fused_{i+1:04d}.jpg")
    cv2.imwrite(save_path, fused_img)

    # Display side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Traditional (testA)")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB))
    axes[1].set_title("AI Painting (testB)")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(fused_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Fused (TCP Preserved)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

print(f"✅ Sample fused images saved in: {fused_folder}")


import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Paths
tcp_path = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/preprocessed_Chinese_Landscape_Painting2photo/testA"
aip_path = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/preprocessed_Chinese_Landscape_Painting2photo/testB"
output_path = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/fused_TCP_preserved"

os.makedirs(output_path, exist_ok=True)

# Resize dimensions
IMG_SIZE = (256, 256)

# Fusion process
for i, img_name in enumerate(tqdm(sorted(os.listdir(tcp_path)))):
    if not img_name.endswith(".jpg") or "_gray" in img_name or "_edges" in img_name or "_hist_eq" in img_name:
        continue

    # Load images
    img_tcp = cv2.imread(os.path.join(tcp_path, img_name))
    img_aip = cv2.imread(os.path.join(aip_path, img_name))  # Must be same-named files

    if img_tcp is None or img_aip is None:
        continue

    img_tcp = cv2.resize(img_tcp, IMG_SIZE)
    img_aip = cv2.resize(img_aip, IMG_SIZE)

    # Get grayscale for structure
    tcp_gray = cv2.cvtColor(img_tcp, cv2.COLOR_BGR2GRAY)
    edge_tcp = cv2.Canny(tcp_gray, 100, 200)

    # Fuzzy weights based on structure (edge strength)
    edge_strength = edge_tcp / 255.0  # normalize
    tcp_weight = 0.7 + 0.3 * (1 - edge_strength)  # stronger structure = more TCP weight
    aip_weight = 1.0 - tcp_weight  # remaining goes to AI color

    # Apply fusion per channel
    fused = np.zeros_like(img_tcp, dtype=np.uint8)
    for c in range(3):
        fused[..., c] = (img_tcp[..., c] * tcp_weight + img_aip[..., c] * aip_weight).astype(np.uint8)

    # Save fused image
    fused_name = f"fused_{i+1:04d}.jpg"
    fused_save_path = os.path.join(output_path, fused_name)
    cv2.imwrite(fused_save_path, fused)

    # Show sample
    if i < 3:  # Show only 3 samples
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(cv2.cvtColor(img_tcp, cv2.COLOR_BGR2RGB))
        axs[0].set_title("TCP (Structure)")
        axs[1].imshow(cv2.cvtColor(img_aip, cv2.COLOR_BGR2RGB))
        axs[1].set_title("AIP (Color)")
        axs[2].imshow(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB))
        axs[2].set_title("Fused Output")
        for ax in axs: ax.axis("off")
        plt.show()

print(f"✅ All fused images saved at: {output_path}")


import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Paths
tcp_path = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/preprocessed_Chinese_Landscape_Painting2photo/testA"
aip_path = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/preprocessed_Chinese_Landscape_Painting2photo/testB"
output_path = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/fused_TCP_preserved"

os.makedirs(output_path, exist_ok=True)

# Resize dimensions
IMG_SIZE = (256, 256)

# Show limit
show_limit = 5
shown = 0

# Fusion Process
for i, img_name in enumerate(tqdm(sorted(os.listdir(tcp_path)))):
    if not img_name.endswith(".jpg") or "_gray" in img_name or "_edges" in img_name or "_hist_eq" in img_name:
        continue

    img_tcp = cv2.imread(os.path.join(tcp_path, img_name))
    img_aip = cv2.imread(os.path.join(aip_path, img_name))

    if img_tcp is None or img_aip is None:
        continue

    img_tcp = cv2.resize(img_tcp, IMG_SIZE)
    img_aip = cv2.resize(img_aip, IMG_SIZE)

    # Grayscale and edge detection
    tcp_gray = cv2.cvtColor(img_tcp, cv2.COLOR_BGR2GRAY)
    edge_tcp = cv2.Canny(tcp_gray, 100, 200)

    edge_strength = edge_tcp / 255.0
    tcp_weight = 0.7 + 0.3 * (1 - edge_strength)
    aip_weight = 1.0 - tcp_weight

    fused = np.zeros_like(img_tcp, dtype=np.uint8)
    for c in range(3):
        fused[..., c] = (img_tcp[..., c] * tcp_weight + img_aip[..., c] * aip_weight).astype(np.uint8)

    # Save fused image
    fused_name = f"fused_{i+1:04d}.jpg"
    fused_save_path = os.path.join(output_path, fused_name)
    cv2.imwrite(fused_save_path, fused)

    # 🔍 Show preview for first few images
    if shown < show_limit:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img_tcp, cv2.COLOR_BGR2RGB))
        plt.title("TCP (Structure)")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(img_aip, cv2.COLOR_BGR2RGB))
        plt.title("AIP (Color)")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB))
        plt.title("Fused Output")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
        shown += 1

print(f"✅ All fused images saved at: {output_path}")


import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define paths
tcp_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/preprocessed_Chinese_Landscape_Painting2photo/testA"
ai_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/preprocessed_Chinese_Landscape_Painting2photo/testB"
fused_output = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/fused_TCP_preserved"

# Create output directory
os.makedirs(fused_output, exist_ok=True)

# Get sorted matching image pairs
tcp_images = sorted([img for img in os.listdir(tcp_folder) if img.endswith(".jpg") and "_gray" not in img and "_edges" not in img and "_hist_eq" not in img])
ai_images = sorted([img for img in os.listdir(ai_folder) if img.endswith(".jpg") and "_gray" not in img and "_edges" not in img and "_hist_eq" not in img])
pair_count = min(len(tcp_images), len(ai_images))

# Fuzzy Fusion: structure from TCP, color from AI
for i in tqdm(range(pair_count)):
    img_tcp = cv2.imread(os.path.join(tcp_folder, tcp_images[i]))
    img_ai = cv2.imread(os.path.join(ai_folder, ai_images[i]))

    # Resize just in case
    img_tcp = cv2.resize(img_tcp, (256, 256))
    img_ai = cv2.resize(img_ai, (256, 256))

    # Convert TCP to grayscale to preserve structure
    tcp_gray = cv2.cvtColor(img_tcp, cv2.COLOR_BGR2GRAY)
    tcp_gray_colored = cv2.cvtColor(tcp_gray, cv2.COLOR_GRAY2BGR)

    # Extract color features from AI painting
    ai_hsv = cv2.cvtColor(img_ai, cv2.COLOR_BGR2HSV)
    tcp_hsv = cv2.cvtColor(tcp_gray_colored, cv2.COLOR_BGR2HSV)

    # Fuzzy-inspired fusion: Use AI color (H and S) with TCP structure (V)
    fused_hsv = np.zeros_like(tcp_hsv)
    fused_hsv[:, :, 0] = ai_hsv[:, :, 0]  # Hue from AI
    fused_hsv[:, :, 1] = ai_hsv[:, :, 1]  # Saturation from AI
    fused_hsv[:, :, 2] = tcp_hsv[:, :, 2]  # Value (brightness) from TCP

    fused_img = cv2.cvtColor(fused_hsv, cv2.COLOR_HSV2BGR)

    # Save image
    filename = f"fused_{i+1:04d}.jpg"
    cv2.imwrite(os.path.join(fused_output, filename), fused_img)

    # Show first 3 samples in Colab
    if i < 3:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(cv2.cvtColor(img_tcp, cv2.COLOR_BGR2RGB))
        axs[0].set_title("TCP Input")
        axs[0].axis("off")

        axs[1].imshow(cv2.cvtColor(img_ai, cv2.COLOR_BGR2RGB))
        axs[1].set_title("AI Painting")
        axs[1].axis("off")

        axs[2].imshow(cv2.cvtColor(fused_img, cv2.COLOR_BGR2RGB))
        axs[2].set_title("Fused Output")
        axs[2].axis("off")

        plt.tight_layout()
        plt.show()

print(f"✅ All fused images saved at: {fused_output}")


from google.colab import drive
drive.mount('/content/drive')

!pip install ImageHash


import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import imagehash
from shutil import copy2
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

# Set your dataset paths
base_path = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/Chinese_Landscape_Painting2photo"
output_phash = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/1to1_phash_matched"

os.makedirs(output_phash, exist_ok=True)

def phash(img_path):
    try:
        img = Image.open(img_path).convert("L").resize((64, 64))
        return imagehash.phash(img)
    except:
        return None

def match_and_copy(folderA, folderB, outA, outB, sample=False):
    pathA = os.path.join(base_path, folderA)
    pathB = os.path.join(base_path, folderB)
    outA_path = os.path.join(output_phash, outA)
    outB_path = os.path.join(output_phash, outB)
    os.makedirs(outA_path, exist_ok=True)
    os.makedirs(outB_path, exist_ok=True)

    filesA = sorted(os.listdir(pathA))
    filesB = sorted(os.listdir(pathB))

    print(f"🔍 Matching {folderA} ↔ {folderB} using pHash...")

    phashes_B = {f: phash(os.path.join(pathB, f)) for f in tqdm(filesB)}

    for fnameA in tqdm(filesA):
        path_imgA = os.path.join(pathA, fnameA)
        hashA = phash(path_imgA)
        if hashA is None:
            continue
        best_match = min(phashes_B.items(), key=lambda x: (hashA - x[1]) if x[1] else 999)[0]
        copy2(path_imgA, os.path.join(outA_path, fnameA))
        copy2(os.path.join(pathB, best_match), os.path.join(outB_path, fnameA))  # Rename as same

    print(f"✅ Matching completed: {folderA} → {outA}, {folderB} → {outB}")

    if sample:
        # Show first few pairs
        fig, axs = plt.subplots(3, 2, figsize=(10, 8))
        for i, img_name in enumerate(sorted(os.listdir(outA_path))[:3]):
            img1 = cv2.imread(os.path.join(outA_path, img_name))
            img2 = cv2.imread(os.path.join(outB_path, img_name))
            axs[i, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            axs[i, 0].set_title(f"TCP - {img_name}")
            axs[i, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            axs[i, 1].set_title(f"AI - {img_name}")
        plt.tight_layout()
        plt.show()

# Run for all sets
match_and_copy("trainA", "trainB", "trainA_phash", "trainB_phash", sample=True)
match_and_copy("testA", "testB", "testA_phash", "testB_phash", sample=True)


!ls "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset"


!ls "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/1to1_phash_matched"


import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Correct paths based on your folder names
tcp_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/1to1_phash_matched/testA_phash"
ai_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/1to1_phash_matched/testB_phash"
output_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/fused_TCP_preserved_phash"

os.makedirs(output_folder, exist_ok=True)

# Load image names
tcp_images = sorted([img for img in os.listdir(tcp_folder) if img.endswith(".jpg")])
ai_images = sorted([img for img in os.listdir(ai_folder) if img.endswith(".jpg")])

assert len(tcp_images) == len(ai_images), "Mismatch in number of testA_phash and testB_phash images!"

# Fuzzy fusion function
def fuzzy_fuse(imgA, imgB):
    imgA = cv2.resize(imgA, (256, 256))
    imgB = cv2.resize(imgB, (256, 256))

    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    edgeA = cv2.Canny(grayA, 100, 200)
    edgeB = cv2.Canny(grayB, 100, 200)

    contrastA = np.var(grayA)
    contrastB = np.var(grayB)

    edge_meanA = np.mean(edgeA)
    edge_meanB = np.mean(edgeB)

    # Fuzzy weights
    contrast_weight = contrastA / (contrastA + contrastB + 1e-5)
    edge_weight = edge_meanB / (edge_meanA + edge_meanB + 1e-5)

    tcp_weight = 0.6 * contrast_weight + 0.4 * (1 - edge_weight)
    ai_weight = 1 - tcp_weight

    # Fuse image preserving TCP
    fused = cv2.addWeighted(imgA, tcp_weight, imgB, ai_weight, 0)
    return fused

# Fuse and display
for i in tqdm(range(len(tcp_images))):
    pathA = os.path.join(tcp_folder, tcp_images[i])
    pathB = os.path.join(ai_folder, ai_images[i])

    imgA = cv2.imread(pathA)
    imgB = cv2.imread(pathB)

    if imgA is None or imgB is None:
        continue

    fused = fuzzy_fuse(imgA, imgB)

    # Save output
    fused_path = os.path.join(output_folder, f"fused_{i+1:04d}.jpg")
    cv2.imwrite(fused_path, fused)

    # Show every 200th image
    if i % 200 == 0:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB))
        plt.title("TCP Input")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB))
        plt.title("AI Input")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB))
        plt.title("Fused Output")
        plt.axis("off")
        plt.show()

print(f"\n✅ All fused images saved at: {output_folder}")


import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Paths
tcp_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/1to1_phash_matched/testA_phash"
ai_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/1to1_phash_matched/testB_phash"
fused_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/fused_TCP_preserved_phash"

# Collect image names
tcp_images = sorted([img for img in os.listdir(tcp_folder) if img.endswith(".jpg")])
ai_images = sorted([img for img in os.listdir(ai_folder) if img.endswith(".jpg")])
fused_images = sorted([img for img in os.listdir(fused_folder) if img.endswith(".jpg")])

# Match by SSIM
ssim_scores = []
for i in tqdm(range(len(tcp_images))):
    tcp_img = cv2.imread(os.path.join(tcp_folder, tcp_images[i]), cv2.IMREAD_GRAYSCALE)
    ai_img = cv2.imread(os.path.join(ai_folder, ai_images[i]), cv2.IMREAD_GRAYSCALE)

    if tcp_img is None or ai_img is None:
        continue

    tcp_img = cv2.resize(tcp_img, (256, 256))
    ai_img = cv2.resize(ai_img, (256, 256))

    score, _ = ssim(tcp_img, ai_img, full=True)
    ssim_scores.append((i, score))

# Sort by highest similarity
top_matches = sorted(ssim_scores, key=lambda x: x[1], reverse=True)[:5]  # Top 5

# Show results
for idx, score in top_matches:
    tcp = cv2.imread(os.path.join(tcp_folder, tcp_images[idx]))
    ai = cv2.imread(os.path.join(ai_folder, ai_images[idx]))
    fused = cv2.imread(os.path.join(fused_folder, fused_images[idx]))

    tcp = cv2.cvtColor(cv2.resize(tcp, (256, 256)), cv2.COLOR_BGR2RGB)
    ai = cv2.cvtColor(cv2.resize(ai, (256, 256)), cv2.COLOR_BGR2RGB)
    fused = cv2.cvtColor(cv2.resize(fused, (256, 256)), cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(tcp)
    plt.title(f"TCP (SSIM: {score:.2f})")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(ai)
    plt.title("AI Painting")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(fused)
    plt.title("Fused Output")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


import os
import cv2
import matplotlib.pyplot as plt

# Update paths to your Google Drive folders
tcp_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/1to1_phash_matched/testA_phash"
ai_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/1to1_phash_matched/testB_phash"
fused_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/fused_TCP_preserved_phash"

# Get sorted image names
tcp_images = sorted([img for img in os.listdir(tcp_folder) if img.endswith(".jpg")])
ai_images = sorted([img for img in os.listdir(ai_folder) if img.endswith(".jpg")])
fused_images = sorted([img for img in os.listdir(fused_folder) if img.endswith(".jpg")])

# Limit how many you want to display
num_samples = 15  # You can change this to 10, 20, etc.

for idx in range(num_samples):
    tcp_img = cv2.imread(os.path.join(tcp_folder, tcp_images[idx]))
    ai_img = cv2.imread(os.path.join(ai_folder, ai_images[idx]))
    fused_img = cv2.imread(os.path.join(fused_folder, fused_images[idx]))

    # Convert BGR to RGB for display
    tcp_img = cv2.cvtColor(tcp_img, cv2.COLOR_BGR2RGB)
    ai_img = cv2.cvtColor(ai_img, cv2.COLOR_BGR2RGB)
    fused_img = cv2.cvtColor(fused_img, cv2.COLOR_BGR2RGB)

    # Plot images
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(tcp_img)
    plt.title("🎨 TCP (Traditional)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(ai_img)
    plt.title("🧠 AI Painting")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(fused_img)
    plt.title("🔀 Fused Output")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


from google.colab import drive
drive.mount('/content/drive')

import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

# Your paths
tcp_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/1to1_phash_matched/testA_phash"
fused_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/fused_TCP_preserved_phash"

# List images
tcp_images = sorted([f for f in os.listdir(tcp_folder) if f.endswith(".jpg")])
fused_images = sorted([f for f in os.listdir(fused_folder) if f.endswith(".jpg")])

# SSIM calculation
ssim_scores = []

for tcp_img_name, fused_img_name in zip(tcp_images, fused_images):
    tcp_img = cv2.imread(os.path.join(tcp_folder, tcp_img_name), cv2.IMREAD_GRAYSCALE)
    fused_img = cv2.imread(os.path.join(fused_folder, fused_img_name), cv2.IMREAD_GRAYSCALE)

    if tcp_img is None or fused_img is None:
        continue

    tcp_img = cv2.resize(tcp_img, (256, 256))
    fused_img = cv2.resize(fused_img, (256, 256))

    score, _ = ssim(tcp_img, fused_img, full=True)
    ssim_scores.append(score)

avg_ssim = np.mean(ssim_scores)
print(f"✅ Average SSIM between TCP and Fused Images: {avg_ssim:.4f}")


import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define Paths
tcp_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/1to1_phash_matched/testA_phash"
ai_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/1to1_phash_matched/testB_phash"
output_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/fused_TCP_preserved_phash_improved"

os.makedirs(output_folder, exist_ok=True)

# Load image names
tcp_images = sorted([img for img in os.listdir(tcp_folder) if img.endswith(".jpg")])
ai_images = sorted([img for img in os.listdir(ai_folder) if img.endswith(".jpg")])

# Improved Fusion
for idx, (tcp_img_name, ai_img_name) in tqdm(enumerate(zip(tcp_images, ai_images)), total=len(tcp_images)):

    # Load images
    tcp_img_path = os.path.join(tcp_folder, tcp_img_name)
    ai_img_path = os.path.join(ai_folder, ai_img_name)

    tcp_img = cv2.imread(tcp_img_path)
    ai_img = cv2.imread(ai_img_path)

    if tcp_img is None or ai_img is None:
        continue

    # Resize if mismatch
    if tcp_img.shape != ai_img.shape:
        ai_img = cv2.resize(ai_img, (tcp_img.shape[1], tcp_img.shape[0]))

    # Improved Static Weights (fixed weights)
    TCP_WEIGHT = 0.75  # Increase TCP contribution
    AI_WEIGHT = 0.25   # Decrease AI contribution

    fused = cv2.addWeighted(tcp_img, TCP_WEIGHT, ai_img, AI_WEIGHT, 0)

    # Save Fused Image
    fused_name = f"fused_{idx+1:04d}.jpg"
    fused_path = os.path.join(output_folder, fused_name)
    cv2.imwrite(fused_path, fused)

print(f"✅ All improved fused images saved at: {output_folder}")

# Display some fused samples
sample_indices = np.random.choice(len(tcp_images), size=5, replace=False)

for idx in sample_indices:
    tcp_img = cv2.imread(os.path.join(tcp_folder, tcp_images[idx]))
    ai_img = cv2.imread(os.path.join(ai_folder, ai_images[idx]))
    fused_img = cv2.imread(os.path.join(output_folder, f"fused_{idx+1:04d}.jpg"))

    # Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(tcp_img, cv2.COLOR_BGR2RGB))
    plt.title("TCP Painting")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(ai_img, cv2.COLOR_BGR2RGB))
    plt.title("AI Painting")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(fused_img, cv2.COLOR_BGR2RGB))
    plt.title("Improved Fused Output")
    plt.axis('off')

    plt.show()


import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# Paths
tcp_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/1to1_phash_matched/testA_phash"
fused_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/fused_TCP_preserved_phash_improved"

# List images
tcp_images = sorted([img for img in os.listdir(tcp_folder) if img.endswith(".jpg")])
fused_images = sorted([img for img in os.listdir(fused_folder) if img.endswith(".jpg")])

# Calculate SSIM
ssim_scores = []

for idx in tqdm(range(len(tcp_images))):
    tcp_img_path = os.path.join(tcp_folder, tcp_images[idx])
    fused_img_path = os.path.join(fused_folder, fused_images[idx])

    tcp_img = cv2.imread(tcp_img_path, cv2.IMREAD_GRAYSCALE)
    fused_img = cv2.imread(fused_img_path, cv2.IMREAD_GRAYSCALE)

    if tcp_img is None or fused_img is None:
        continue

    if tcp_img.shape != fused_img.shape:
        fused_img = cv2.resize(fused_img, (tcp_img.shape[1], tcp_img.shape[0]))

    score = ssim(tcp_img, fused_img)
    ssim_scores.append(score)

# Average SSIM
average_ssim = np.mean(ssim_scores)
print(f"✅ Average SSIM between TCP and Improved Fused Images: {average_ssim:.4f}")


import matplotlib.pyplot as plt

# Assuming ssim_scores list already calculated from previous step

# Plot the histogram
plt.figure(figsize=(8,5))
plt.hist(ssim_scores, bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of SSIM Scores (TCP vs Improved Fused Images)', fontsize=14)
plt.xlabel('SSIM Score', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.grid(True)
plt.show()


import random
import matplotlib.pyplot as plt
import cv2
import os

# Define folders again
tcp_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/1to1_phash_matched/testA_phash"
fused_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/fused_TCP_preserved_phash_improved"

# Randomly select 5 images
sample_indices = random.sample(range(len(os.listdir(tcp_folder))), 5)
tcp_images = sorted([img for img in os.listdir(tcp_folder) if img.endswith(".jpg")])
fused_images = sorted([img for img in os.listdir(fused_folder) if img.endswith(".jpg")])

# Plot samples
plt.figure(figsize=(15,10))
for i, idx in enumerate(sample_indices):
    tcp_img_path = os.path.join(tcp_folder, tcp_images[idx])
    fused_img_path = os.path.join(fused_folder, fused_images[idx])

    tcp_img = cv2.imread(tcp_img_path)
    tcp_img = cv2.cvtColor(tcp_img, cv2.COLOR_BGR2RGB)

    fused_img = cv2.imread(fused_img_path)
    fused_img = cv2.cvtColor(fused_img, cv2.COLOR_BGR2RGB)

    plt.subplot(5, 2, 2*i+1)
    plt.imshow(tcp_img)
    plt.title(f"Original TCP {idx+1}")
    plt.axis('off')

    plt.subplot(5, 2, 2*i+2)
    plt.imshow(fused_img)
    plt.title(f"Fused Image {idx+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()


from google.colab import drive
drive.mount('/content/drive')

# Install BLIP Model
!pip install transformers
!pip install timm
!pip install accelerate

# Import Libraries
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

# Load BLIP Model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu")

# Paths
fused_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/fused_TCP_preserved_phash_improved"
output_csv_path = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/fused_TCP_preserved_phash_improved/semantic_analysis_results.csv"

# Prepare Storage
results = []

# Generate Meaning for Each Image
for img_name in tqdm(sorted(os.listdir(fused_folder))):
    if img_name.endswith(".jpg"):
        img_path = os.path.join(fused_folder, img_name)

        raw_image = Image.open(img_path).convert('RGB')

        inputs = processor(raw_image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        out = model.generate(**inputs)

        caption = processor.decode(out[0], skip_special_tokens=True)

        results.append({"Image": img_name, "Meaning": caption})

# Save Results
df = pd.DataFrame(results)
df.to_csv(output_csv_path, index=False)

print(f"✅ Semantic Analysis Completed. Results saved at: {output_csv_path}")


import pandas as pd

# Load your semantic CSV
semantic_csv = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/fused_TCP_preserved_phash_improved/semantic_analysis_results.csv"
semantic_results = pd.read_csv(semantic_csv)

# See available columns
print("✅ Available Columns in CSV:", list(semantic_results.columns))
semantic_results.head()


import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

# Paths
fused_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/fused_TCP_preserved_phash_improved"
semantic_csv = os.path.join(fused_folder, "semantic_analysis_results.csv")

# Load the semantic results
semantic_results = pd.read_csv(semantic_csv)

# Display 10 random fused images along with their meanings
num_samples = 10

plt.figure(figsize=(15, 20))

for idx in range(num_samples):
    # Get image name and meaning
    img_name = semantic_results.iloc[idx]['Image']
    meaning = semantic_results.iloc[idx]['Meaning']

    # Load the image
    img_path = os.path.join(fused_folder, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Plot
    plt.subplot(5, 2, idx+1)
    plt.imshow(img)
    plt.title(f"Meaning: {meaning}", fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()


from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

# Paths
fused_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/fused_TCP_preserved_phash_improved"
semantic_csv = os.path.join(fused_folder, "semantic_analysis_results.csv")
save_path = os.path.join(fused_folder, "semantic_visualization_output.png")  # ✅ Output image file

# Load the semantic results
semantic_results = pd.read_csv(semantic_csv)

# Display 10 random fused images along with their meanings
num_samples = 10

plt.figure(figsize=(15, 20))

for idx in range(num_samples):
    img_name = semantic_results.iloc[idx]['Image']
    meaning = semantic_results.iloc[idx]['Meaning']

    # Load and convert image
    img_path = os.path.join(fused_folder, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Plot
    plt.subplot(5, 2, idx + 1)
    plt.imshow(img)
    plt.title(f"Meaning: {meaning}", fontsize=10)
    plt.axis('off')

plt.tight_layout()

# ✅ Save the figure
plt.savefig(save_path)
print(f"✅ Output saved at: {save_path}")

# Also display
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

# Paths
fused_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/fused_TCP_preserved_phash_improved"
semantic_csv = os.path.join(fused_folder, "semantic_analysis_results.csv")

# ✅ New Save Path in Extracted_Dataset root
save_path = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/semantic_visualization_output.png"

# Load the semantic results
semantic_results = pd.read_csv(semantic_csv)

# Display 10 random fused images along with their meanings
num_samples = 10

plt.figure(figsize=(15, 20))

for idx in range(num_samples):
    img_name = semantic_results.iloc[idx]['Image']
    meaning = semantic_results.iloc[idx]['Meaning']

    # Load and convert image
    img_path = os.path.join(fused_folder, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Plot
    plt.subplot(5, 2, idx + 1)
    plt.imshow(img)
    plt.title(f"Meaning: {meaning}", fontsize=10)
    plt.axis('off')

plt.tight_layout()

# ✅ Save the figure to Extracted_Dataset
plt.savefig(save_path)
print(f"✅ Output saved at: {save_path}")

# Also display
plt.show()


from google.colab import drive
drive.mount('/content/drive')

# Install necessary libraries
!pip install -q transformers timm accelerate

# Import required libraries
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda" if torch.cuda.is_available() else "cpu")

# Define folders
fused_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/fused_TCP_preserved_phash"  # Your fused output images
poetic_captions_save_path = "/content/drive/My Drive/Colab Notebooks/Colab_outputs/poetic_meanings.csv"
os.makedirs(os.path.dirname(poetic_captions_save_path), exist_ok=True)

# Generate captions with poetic prompt
poetic_descriptions = []
image_files = sorted([img for img in os.listdir(fused_folder) if img.endswith(".jpg")])

for img_name in tqdm(image_files, desc="Generating poetic meanings"):
    img_path = os.path.join(fused_folder, img_name)
    raw_image = Image.open(img_path).convert('RGB')

    # Give a poetic prompt
    poetic_prompt = "Describe this image as a traditional Chinese painting with emotions and scenery details."

    inputs = processor(raw_image, poetic_prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)

    poetic_descriptions.append({"Image": img_name, "Meaning": caption})

# Save results
df = pd.DataFrame(poetic_descriptions)
df.to_csv(poetic_captions_save_path, index=False)

print(f"✅ Poetic meanings saved at: {poetic_captions_save_path}")

# Display a few sample images with meanings
df_sample = df.sample(10, random_state=42)

fig, axs = plt.subplots(5, 2, figsize=(12, 24))
axs = axs.flatten()

for idx, (i, row) in enumerate(df_sample.iterrows()):
    img_path = os.path.join(fused_folder, row['Image'])
    img = Image.open(img_path)

    axs[idx].imshow(img)
    axs[idx].axis('off')
    axs[idx].set_title(f"Meaning: {row['Meaning']}", fontsize=10)

plt.tight_layout()
plt.show()

from google.colab import drive
drive.mount('/content/drive')

!ls "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/1to1_phash_matched"


from google.colab import drive
drive.mount('/content/drive')

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define paths (update if needed)
tcp_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/1to1_phash_matched/testA_phash"
ai_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/1to1_phash_matched/testB_phash"

# Load image filenames
tcp_images = sorted([f for f in os.listdir(tcp_folder) if f.endswith(".jpg")])
ai_images = sorted([f for f in os.listdir(ai_folder) if f.endswith(".jpg")])

# Ensure matching count
assert len(tcp_images) == len(ai_images), "Mismatch between TCP and AI image counts!"

# Choose 5 random samples to show in Colab
sample_indices = np.random.choice(len(tcp_images), size=5, replace=False)

TCP_WEIGHT = 0.8
AI_WEIGHT = 0.2

for idx in sample_indices:
    tcp_img_path = os.path.join(tcp_folder, tcp_images[idx])
    ai_img_path = os.path.join(ai_folder, ai_images[idx])

    tcp_img = cv2.imread(tcp_img_path)
    ai_img = cv2.imread(ai_img_path)

    if tcp_img is None or ai_img is None:
        continue

    # Resize if needed
    if tcp_img.shape != ai_img.shape:
        ai_img = cv2.resize(ai_img, (tcp_img.shape[1], tcp_img.shape[0]))

    # Fuse images using 80% TCP and 20% AI
    fused = cv2.addWeighted(tcp_img, TCP_WEIGHT, ai_img, AI_WEIGHT, 0)

    # Display in Colab
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(tcp_img, cv2.COLOR_BGR2RGB))
    plt.title("TCP Painting (80%)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(ai_img, cv2.COLOR_BGR2RGB))
    plt.title("AI Painting (20%)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB))
    plt.title("Fused Output")
    plt.axis('off')

    plt.show()


import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# Define paths
tcp_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/1to1_phash_matched/testA_phash"
ai_folder = "/content/drive/My Drive/Colab Notebooks/Extracted_Dataset/1to1_phash_matched/testB_phash"

# Load image filenames
tcp_images = sorted([f for f in os.listdir(tcp_folder) if f.endswith(".jpg")])
ai_images = sorted([f for f in os.listdir(ai_folder) if f.endswith(".jpg")])

# Make sure images are aligned
assert len(tcp_images) == len(ai_images), "Mismatch between TCP and AI images!"

TCP_WEIGHT = 0.8
AI_WEIGHT = 0.2

ssim_scores = []

for i in tqdm(range(len(tcp_images))):
    tcp_path = os.path.join(tcp_folder, tcp_images[i])
    ai_path = os.path.join(ai_folder, ai_images[i])

    tcp_img = cv2.imread(tcp_path)
    ai_img = cv2.imread(ai_path)

    if tcp_img is None or ai_img is None:
        continue

    # Resize if shapes don't match
    if tcp_img.shape != ai_img.shape:
        ai_img = cv2.resize(ai_img, (tcp_img.shape[1], tcp_img.shape[0]))

    # Fuse the images (80% TCP, 20% AI)
    fused_img = cv2.addWeighted(tcp_img, TCP_WEIGHT, ai_img, AI_WEIGHT, 0)

    # Convert to grayscale
    gray_tcp = cv2.cvtColor(tcp_img, cv2.COLOR_BGR2GRAY)
    gray_fused = cv2.cvtColor(fused_img, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM
    score = ssim(gray_tcp, gray_fused)
    ssim_scores.append(score)

# Final SSIM Report
average_ssim = np.mean(ssim_scores)
print(f"\n✅ Average SSIM between TCP and Fused Images (80%/20%): {average_ssim:.4f}")
