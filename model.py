import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

dataset_path = "data2"
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")
val_path = os.path.join(dataset_path, "val")

open_train_path = os.path.join(train_path, "open")
close_train_path = os.path.join(train_path, "closed")
open_test_path = os.path.join(test_path, "open")
close_test_path = os.path.join(test_path, "closed")
open_val_path = os.path.join(val_path, "open")
close_val_path = os.path.join(val_path, "closed")

hog_params = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": 'L2-Hys'
}

def process_images(folder_path, label):
    features_list = []
    labels_list = []

    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (32, 32))
        features = hog(img, **hog_params)

        features_list.append(features)
        labels_list.append(label)

    return features_list, labels_list


X_train_open, y_train_open = process_images(open_train_path, 1)
X_train_closed, y_train_closed = process_images(close_train_path, 0)
X_test_open, y_test_open = process_images(open_test_path, 1)
X_test_closed, y_test_closed = process_images(close_test_path, 0)
X_val_open, y_val_open = process_images(open_val_path, 1)
X_val_closed, y_val_closed = process_images(close_val_path, 0)


X_train = np.array(X_train_open + X_train_closed)
y_train = np.array(y_train_open + y_train_closed)
X_test = np.array(X_test_open + X_test_closed)
y_test = np.array(y_test_open + y_test_closed)
X_val = np.array(X_val_open + X_val_closed)
y_val = np.array(y_val_open + y_val_closed)

print(f"Veri seti işlendi! Eğitim: {len(X_train)}, Test: {len(X_test)}, Validation: {len(X_val)}")

log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train, y_train)


y_train_pred = log_reg_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Eğitim doğruluğu: {train_accuracy * 100:.2f}%")

y_pred = log_reg_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test doğruluğu: {test_accuracy * 100:.2f}%")


y_val_pred = log_reg_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation doğruluğu: {val_accuracy * 100:.2f}%")


joblib.dump(log_reg_model, 'logistic_regression_model.pkl')
print("Model kaydedildi!")


