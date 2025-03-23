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

