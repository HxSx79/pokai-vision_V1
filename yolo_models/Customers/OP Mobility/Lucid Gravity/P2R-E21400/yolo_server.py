# yolo_server.py
#
# Description:
# This script creates a high-performance inference server using ZeroMQ.
# It loads a YOLOv8 .pt model and listens for image data from a client (like our Qt app).
# It now performs inference and sends back a JSON object containing an array of all
# detected objects, including their class, confidence, and bounding box.
#
# Prerequisites:
#   pip install ultralytics pyzmq numpy
#
# Usage:
# 1. Place this script in a folder with your 'best.pt' and 'obj.names' files.
# 2. Run from the terminal: python yolo_server.py
# 3. The server will print "YOLO Inference Server is running..." and wait for connections.

import zmq
import numpy as np
import cv2
import json
from ultralytics import YOLO

# --- Configuration ---
PT_MODEL_PATH = 'best.pt'
NAMES_FILE_PATH = 'obj.names'
ZMQ_PORT = "5555"
CONF_THRESHOLD = 0.5

def main():
    """ Main function to run the YOLO inference server """
    
    # --- Load YOLO Model and Class Names ---
    try:
        print(f"Loading YOLO model from: {PT_MODEL_PATH}")
        model = YOLO(PT_MODEL_PATH)
        print("Model loaded successfully.")
        
        with open(NAMES_FILE_PATH, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(classes)} class names: {classes}")

    except Exception as e:
        print(f"Error loading model or class names: {e}")
        return

    # --- Setup ZeroMQ Socket ---
    context = zmq.Context()
    socket = context.socket(zmq.REP) # REP for Reply socket
    socket.bind(f"tcp://*:{ZMQ_PORT}")
    print(f"YOLO Inference Server is running on port {ZMQ_PORT}...")

    # --- Main Server Loop ---
    while True:
        try:
            # Wait for a request from the client
            image_bytes = socket.recv()

            # Decode the image bytes into an OpenCV frame
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                raise ValueError("Failed to decode image")

            # --- Perform Inference ---
            results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
            
            # --- Prepare Detection Data ---
            detections = []
            
            # The first result object contains the detections for our single image
            result = results[0]
            
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detection_data = {
                    'class_id': class_id,
                    'class_name': classes[class_id] if class_id < len(classes) else 'Unknown',
                    'confidence': confidence,
                    'box': [x1, y1, x2, y2]
                }
                detections.append(detection_data)

            # --- Prepare and Send JSON Response ---
            # The response is now a dictionary containing a list of detections
            response = {
                "detections": detections
            }
            
            socket.send_json(response)

        except Exception as e:
            print(f"An error occurred during processing: {e}")
            # Send an error response back to the client
            error_response = {"detections": []}
            socket.send_json(error_response)


if __name__ == "__main__":
    main()

