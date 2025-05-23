Open/Closed Eye Detection

This project detects whether a person's eyes are open or closed using a pre-trained model.
The model has already been trained and saved, so you only need to install the required libraries and run the camera script to use it.

How It Works

The system uses OpenCV to capture frames from the webcam and detect the eye region. 
Then, Histogram of Oriented Gradients (HOG) features are extracted and fed into a pre-trained logistic regression model to determine whether the eyes are open or closed.

Technologies Used
- Python
- OpenCV
- Scikit-learn
- NumPy
- Matplotlib

Dataset

The model was trained using the data2 dataset, which consists of three main folders:

- train/ - Contains training images categorized into open/ and closed/ subfolders.

- test/ - Contains test images used to evaluate model performance.

- val/ - Contains validation images for fine-tuning the model.

Installation & Running the Project

Make sure you're in the project directory, then install the required Python packages by running:

```bash
pip install opencv-python scikit-learn scikit-image numpy joblib

The pre-trained model (logistic_regression_model.pkl) is already included in the project. You can directly start the real-time detection by running:
   python camera.py

The system will access your webcam, detect eyes in real time, and display whether they are open or closed on the screen.
 Press q to quit the application.

 If the webcam doesn’t start, try changing cv2.VideoCapture(1) to cv2.VideoCapture(0) in camera.py.
