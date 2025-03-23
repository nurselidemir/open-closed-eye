import cv2
import joblib
from skimage.feature import hog

hog_params = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": 'L2-Hys'
}

model = joblib.load('logistic_regression_model.pkl')

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(1)
print("Kamera başlatılıyor. 2 saniye bekleyin...")
cv2.waitKey(2000)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı!")
        break


    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if len(eyes) == 0:
        cv2.putText(frame, "Closed", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        for (x, y, w, h) in eyes:
            eye_region = gray[y:y + h, x:x + w]
            eye_resized = cv2.resize(eye_region, (32, 32))
            features = hog(eye_resized, **hog_params).reshape(1, -1)

            prediction = model.predict(features)
            label = "Open" if prediction == 1 else "Closed"

            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


    cv2.imshow("Eye State", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

