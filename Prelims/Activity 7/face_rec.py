import cv2
import numpy as np
import os

# Function to load images and labels
def load_images_from_folder(folder_path):
    X, y = [], []
    label_dict = {}  # Map names to numeric labels
    label_id = 0

    for person_name in os.listdir(folder_path):
        person_path = os.path.join(folder_path, person_name)
        if not os.path.isdir(person_path):
            continue
        
        if person_name not in label_dict:
            label_dict[person_name] = label_id
            label_id += 1

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (200, 200))  # Resize for consistency
            X.append(img)
            y.append(label_dict[person_name])
    
    return X, np.array(y), label_dict

# Load training data
train_path = "images_train"
X_train, y_train, label_dict = load_images_from_folder(train_path)

# Train the LBPH Face Recognizer
model = cv2.face.LBPHFaceRecognizer_create()
model.train(X_train, y_train)

# Save the trained model
model.save("face_model.yml")
np.save("label_dict.npy", label_dict)
print("Model trained and saved successfully!")
