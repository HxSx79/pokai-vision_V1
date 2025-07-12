import zmq
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import json
import configparser
from pathlib import Path

class InferenceServer:
    def __init__(self):
        # --- Read from config file ---
        config = configparser.ConfigParser()
        config.read('config.ini')

        # Get server address and parse the port
        server_address = config.get('ZMQ', 'server_address', fallback='tcp://*:5555')
        self.port = server_address.split(':')[-1]
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://*:{self.port}")
        print(f"Inference server listening on port {self.port} with ROUTER socket")
        
        self.model = None
        self.model_path = ""
        
        default_conf_percent = config.getint('General', 'default_confidence', fallback=80)
        self.confidence_threshold = default_conf_percent / 100.0
        
        # --- START: Add attributes for new detection logic ---
        self.expected_clip_qty = 0
        self.main_part_name = None
        self.clip_ok_name = "Clip_OK"
        # --- END: Add attributes ---

        print("Server started. Waiting for a model load request from the client.")
        print(f"Initial confidence threshold set to {self.confidence_threshold}")

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            print(f"[Error] Model path does not exist: {model_path}")
            return False
        try:
            print(f"Loading new model from: {model_path}...")
            self.model = YOLO(model_path)
            
            try:
                self.model.to('cuda')
                print("Model loaded successfully to GPU.")
            except Exception as e:
                print(f"[Warning] Failed to load model to GPU: {e}. Falling back to CPU.")
                self.model.to('cpu')

            self.model_path = model_path
            model_dir = Path(self.model_path).parent
            
            # --- Load object names and quantities ---
            if self.model.names:
                self.main_part_name = self.model.names[0]
                print(f"Main part for detection logic set to: '{self.main_part_name}'")
            else:
                self.main_part_name = None
                print("[Warning] Model has no class names defined.")

            qty_file = model_dir / 'obj.qty'
            if qty_file.is_file():
                try:
                    self.expected_clip_qty = int(qty_file.read_text().strip())
                    print(f"Expected 'Clip_OK' quantity set to: {self.expected_clip_qty}")
                except (ValueError, TypeError):
                    self.expected_clip_qty = 0
                    print(f"[Warning] Could not read '{qty_file}'. Defaulting to 0.")
            else:
                self.expected_clip_qty = 0
                print(f"[Warning] '{qty_file}' not found. Defaulting to 0 expected clips.")
            
            # --- START: Load settings from model_settings.ini ---
            model_settings_path = model_dir / 'model_settings.ini'
            if model_settings_path.is_file():
                print(f"Found model_settings.ini at {model_settings_path}")
                model_config = configparser.ConfigParser()
                model_config.read(model_settings_path)
                
                if model_config.has_option('Confidence', 'threshold'):
                    conf_percent = model_config.getint('Confidence', 'threshold')
                    self.confidence_threshold = conf_percent / 100.0
                    print(f"Loaded confidence from model settings: {conf_percent}%")
                else:
                    print("No confidence setting in model_settings.ini, using default.")
            else:
                print("No model_settings.ini found, using default settings.")
            # --- END: Load settings ---

            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None
            self.model_path = ""
            return False

    def run(self):
        """ The main loop that listens for requests and performs inference. """
        while True:
            try:
                identity, message = self.socket.recv_multipart()

                try:
                    request_str = message.decode('utf-8')
                    if request_str.startswith("LOAD_MODEL::"):
                        new_model_path = request_str.split("::", 1)[1]
                        success = self.load_model(new_model_path)
                        response = {"status": "success" if success else "error"}
                        self.socket.send_multipart([identity, json.dumps(response).encode('utf-8')])
                        continue
                    elif request_str.startswith("SET_CONFIDENCE::"):
                        new_conf = float(request_str.split("::", 1)[1])
                        self.confidence_threshold = new_conf
                        response = {"status": "success"}
                        self.socket.send_multipart([identity, json.dumps(response).encode('utf-8')])
                        continue
                except UnicodeDecodeError:
                    pass
                
                if self.model is None:
                    response = {"status": "error", "message": "No model loaded."}
                    self.socket.send_multipart([identity, json.dumps(response).encode('utf-8')])
                    continue

                np_arr = np.frombuffer(message, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is None:
                    response = {"status": "error", "message": "Could not decode image."}
                    self.socket.send_multipart([identity, json.dumps(response).encode('utf-8')])
                    continue
                
                start_time = time.time()
                results = self.model(img, conf=self.confidence_threshold, verbose=False, half=True, imgsz=640)
                end_time = time.time()
                inference_ms = (end_time - start_time) * 1000
                
                detections_list = []
                part_detected_count = 0
                clips_ok_count = 0
                
                for result in results:
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = self.model.names[cls_id]
                        
                        if cls_name == self.main_part_name:
                            part_detected_count += 1
                        elif cls_name == self.clip_ok_name:
                            clips_ok_count += 1
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections_list.append({
                            "box": [x1, y1, x2, y2],
                            "label": f'{cls_name} {conf:.2f}'
                        })

                part_status = 'PASS' if part_detected_count > 0 else 'FAIL'
                clip_status = 'PASS' if clips_ok_count == self.expected_clip_qty else 'FAIL'

                response_data = {
                    "status": "success",
                    "part_status": part_status,
                    "clip_status": clip_status,
                    "detections": len(detections_list),
                    "inference_ms": inference_ms,
                    "detections_list": detections_list
                }
                
                self.socket.send_multipart([identity, json.dumps(response_data).encode('utf-8')])

            except Exception as e:
                print(f"[Server Error] An error occurred: {e}")

if __name__ == "__main__":
    server = InferenceServer()
    server.run()

