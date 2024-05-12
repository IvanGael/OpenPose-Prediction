import cv2 as cv
import numpy as np
import argparse
import os
from utils import load_model


def calculate_statistics(out, thr):
    total_confidence = 0
    total_detected_parts = 0
    num_parts = out.shape[1]
    for i in range(num_parts):
        heatMap = out[0, i, :, :]
        _, conf, _, _ = cv.minMaxLoc(heatMap)
        if conf > thr:
            total_confidence += conf
            total_detected_parts += 1
    return total_confidence / num_parts, total_detected_parts


# Display Average Confidence Score and Number of Detected Body Parts
def process_frame(net, frame, inWidth, inHeight, BODY_PARTS, POSE_PAIRS, thr):
    frameHeight, frameWidth = frame.shape[:2]
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    avg_confidence, num_detected_parts = calculate_statistics(out, thr)

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thr else None)

    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            # Adding annotation
            cv.putText(frame, partFrom, points[idFrom], cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
            cv.putText(frame, partTo, points[idTo], cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

    # Display average confidence score and number of detected body parts
    cv.putText(frame, f"Avg Confidence: {avg_confidence:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
    cv.putText(frame, f"Detected Parts: {num_detected_parts}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

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
                    cv.imshow('OpenPose using OpenCV', frame)
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



