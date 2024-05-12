
import cv2 as cv
import os


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found.")
    net = cv.dnn.readNetFromTensorflow(model_path)
    return net