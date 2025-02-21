import cv2
import numpy as np
import os

# Load trained model
model = cv2.face.LBPHFaceRecognizer_create()
model.read("face_model.yml")
label_dict = np.load("label_dict.npy", allow_pickle=True).item()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Testing dataset
test_path = "images_test"
correct_recognitions = 0
total_tests = 0

for person_name in os.listdir(test_path):
    person_path = os.path.join(test_path, person_name)
    if not os.path.isdir(person_path):
        continue
    
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (200, 200))

        predicted_label, confidence = model.predict(img)
        predicted_name = [name for name, label in label_dict.items() if label == predicted_label][0]

        if person_name == predicted_name:
            correct_recognitions += 1
        
        total_tests += 1
        print(f"Test Image: {image_name}, Actual: {person_name}, Predicted: {predicted_name}, Confidence: {confidence:.2f}")

accuracy = (correct_recognitions / total_tests) * 100
print(f"\nFinal Accuracy: {accuracy:.2f}% ({correct_recognitions}/{total_tests} correct recognitions)")

# Real-time face recognition
camera = cv2.VideoCapture(0)

while True:
    ret, img = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))

        predicted_label, confidence = model.predict(face_roi)
        predicted_name = [name for name, label in label_dict.items() if label == predicted_label][0]

        cv2.putText(img, f"{predicted_name}, {confidence:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Face Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()