import cv2
import numpy as np

# Load trained model
model = cv2.face.LBPHFaceRecognizer_create()
model.read("face_model.yml")
label_dict = np.load("label_dict.npy", allow_pickle=True).item()

# Print label dictionary for debugging
print("Label Dictionary:", label_dict)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize camera
camera = cv2.VideoCapture(0)

# Confidence threshold for verification
confidence_threshold = 50  # Adjusted threshold

while True:
    ret, img = camera.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI)
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))  # Resize to match training size

        # Predict the label and confidence
        predicted_label, confidence = model.predict(face_roi)

        # Debug: Print predicted label and confidence
        print(f"Predicted Label: {predicted_label}, Confidence: {confidence}")

        # Get the predicted name
        predicted_name = [name for name, label in label_dict.items() if label == predicted_label][0]

        # Debug: Print predicted name
        print(f"Predicted Name: {predicted_name}")

        # Verify if the face is known or unknown
        if confidence < confidence_threshold:
            verification_result = f"Known Face: {predicted_name}"
            color = (0, 255, 0)  # Green for known faces
        else:
            verification_result = "Unknown Face"
            color = (0, 0, 255)  # Red for unknown faces

        # Display the verification result and confidence
        cv2.putText(img, verification_result, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(img, f"Confidence: {confidence:.2f}", (x, y + h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw a rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    # Display the camera feed
    cv2.imshow("Face Recognition", img)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()