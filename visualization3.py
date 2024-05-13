import cv2 as cv
import numpy as np
import argparse
import os
from utils import load_model


# Display the confidence score of each detected body part as text near the corresponding point
def process_frame(net, frame, inWidth, inHeight, BODY_PARTS, POSE_PAIRS, thr):
    frameHeight, frameWidth = frame.shape[:2]
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    confidences = []  # Store confidence scores
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thr else None)
        confidences.append(conf)  # Store confidence score for each body part

    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            # Adding annotation for body parts with confidence score
            if confidences[idFrom] > thr:
                cv.putText(frame, f"{partFrom}: {confidences[idFrom]:.2f}", (points[idFrom][0], points[idFrom][1] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
            if confidences[idTo] > thr:
                cv.putText(frame, f"{partTo}: {confidences[idTo]:.2f}", (points[idTo][0], points[idTo][1] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
    parser.add_argument('--model', default='graph_opt.pb', help='Path to the pre-trained model.')
    parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
    parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
    parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')
    args = parser.parse_args()

    BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                   "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

    POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                   ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                   ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                   ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

    net = load_model(args.model)

    if args.input:
        if os.path.isfile(args.input):  # Check if input is a file
            file_extension = os.path.splitext(args.input)[1].lower()
            if file_extension in ['.jpg', '.jpeg', '.png']:  # If it's an image
                frame = cv.imread(args.input)
                frame = process_frame(net, frame, args.width, args.height, BODY_PARTS, POSE_PAIRS, args.thr)
                cv.imshow('OpenPose', frame)
                cv.waitKey(0)
            elif file_extension in ['.mp4', '.avi', '.mov']:  # If it's a video
                cap = cv.VideoCapture(args.input)
                while cv.waitKey(1) < 0:
                    hasFrame, frame = cap.read()
                    if not hasFrame:
                        cv.waitKey()
                        break
                    frame = process_frame(net, frame, args.width, args.height, BODY_PARTS, POSE_PAIRS, args.thr)
                    cv.imshow('OpenPose', frame)
        else:
            print("Input file not found.")
    else:  # If no input specified, capture frames from camera
        cap = cv.VideoCapture(0)
        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                cv.waitKey()
                break
            frame = process_frame(net, frame, args.width, args.height, BODY_PARTS, POSE_PAIRS, args.thr)
            cv.imshow('OpenPose', frame)




