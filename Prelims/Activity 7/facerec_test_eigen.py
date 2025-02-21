import cv2
import numpy as np
import time

# Load trained model
model = cv2.face.EigenFaceRecognizer_create()
model.read("eigenface_model.yml")
label_dict = np.load("label_dict.npy", allow_pickle=True).item()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, 30)  # Set camera FPS

# Confidence threshold (adjust as needed)
confidence_threshold = 11000

# Store predictions
predictions = []

# Calculate the end time (10 seconds from start)
end_time = time.time() + 10

while time.time() < end_time:
    ret, img = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))

        predicted_label, confidence = model.predict(face_roi)

        if confidence < confidence_threshold:
            predicted_name = [name for name, label in label_dict.items() if label == predicted_label][0]
            verification_result = f"Known Face: {predicted_name}"
            color = (0, 255, 0)
        else:
            predicted_name = "Unknown"
            verification_result = "Unknown Face"
            color = (0, 0, 255)

        # Store result
        predictions.append((predicted_name, confidence))

        # Draw text and rectangle
        cv2.putText(img, verification_result, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(img, f"Confidence: {confidence:.2f}", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Face Recognition", img)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Print captured results
print("Captured Predictions:", predictions[:20])  # Display at least 20 predictions

# Release resources
camera.release()
cv2.destroyAllWindows()
