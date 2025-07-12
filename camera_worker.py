import cv2
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QMutex, QMutexLocker
import configparser
import os
import subprocess
import time
from simple_camera_settings_manager import SimpleCameraSettingsManager

class CameraWorker(QThread):
    """
    A dedicated worker thread to handle camera capture and settings.
    This version includes auto-recovery logic and frame cropping.
    """
    inference_frame_ready = pyqtSignal(object)
    ui_frame_ready = pyqtSignal(object)
    settings_queried = pyqtSignal(dict)
    camera_failed = pyqtSignal()
    camera_ready = pyqtSignal()

    def __init__(self, device_path, parent=None):
        super().__init__(parent)
        self.device_path = device_path
        self.running = True
        self.cap = None
        
        self.failure_count = 0
        self.failure_threshold = 30 

        self.last_applied_settings = {}
        self.settings_manager = SimpleCameraSettingsManager(device_path)
        
        # --- START: Add attributes for cropping ---
        self.crop_mutex = QMutex()
        self.crop_rect = None # Will be a tuple (x, y, w, h)
        # --- END: Add attributes ---

        self.has_v4l2_ctl = self.check_v4l2_ctl()
        if self.has_v4l2_ctl:
            print(f"v4l2-ctl available for {device_path}")
        else:
            print(f"Using OpenCV properties for {device_path}")

    def set_crop(self, rect):
        """Thread-safe method to set the cropping rectangle."""
        with QMutexLocker(self.crop_mutex):
            self.crop_rect = rect
        print(f"Crop set to: {rect}")

    def _crop_frame(self, frame):
        """Applies the current crop rectangle to a frame."""
        with QMutexLocker(self.crop_mutex):
            if self.crop_rect:
                x, y, w, h = self.crop_rect
                return frame[y:y+h, x:x+w]
        return frame

    def check_v4l2_ctl(self):
        """Check if v4l2-ctl is available and device exists"""
        try:
            result = subprocess.run(["which", "v4l2-ctl"], capture_output=True)
            if result.returncode != 0: return False
            if not os.path.exists(self.device_path): return False
            result = subprocess.run(
                ["v4l2-ctl", "-d", self.device_path, "--list-ctrls-menus"],
                capture_output=True, timeout=2
            )
            return result.returncode == 0
        except:
            return False

    def run_v4l2_command(self, cmd):
        """Run a v4l2-ctl command"""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)

    def get_v4l2_control(self, control):
        """Get a v4l2 control value"""
        if not self.has_v4l2_ctl: return None
        success, stdout, stderr = self.run_v4l2_command(f"v4l2-ctl -d {self.device_path} --get-ctrl {control}")
        if success:
            for line in stdout.strip().split('\n'):
                if ':' in line:
                    try: return int(line.split(':')[1].strip())
                    except: return None
        return None

    def set_v4l2_control(self, control, value):
        """Set a v4l2 control using v4l2-ctl"""
        if not self.has_v4l2_ctl: return False
        success, stdout, stderr = self.run_v4l2_command(f"v4l2-ctl -d {self.device_path} --set-ctrl {control}={value}")
        if success:
            print(f"✓ v4l2: Set {control}={value} for {self.device_path}")
            return True
        else:
            print(f"✗ v4l2: Failed to set {control}={value} for {self.device_path}: {stderr}")
            return False

    def get_available_v4l2_controls(self):
        """Get list of available v4l2 controls"""
        if not self.has_v4l2_ctl: return []
        success, stdout, stderr = self.run_v4l2_command(f"v4l2-ctl -d {self.device_path} --list-ctrls")
        if success:
            controls = []
            for line in stdout.split('\n'):
                if any(ctrl in line for ctrl in ['brightness', 'contrast', 'saturation', 'gain', 'sharpness', 'white_balance']):
                    control_name = line.split()[0] if line.strip() else None
                    if control_name: controls.append(control_name)
            return controls
        return []

    @pyqtSlot()
    def query_settings(self):
        """Gets the current values of all supported camera properties using the settings manager."""
        print(f"Querying settings from worker for {self.device_path}...")
        if hasattr(self.settings_manager, 'get_all_controls'):
            settings = self.settings_manager.get_all_controls()
            if settings:
                self.settings_queried.emit(settings)
                return
        settings = {}
        available_controls = self.settings_manager.get_available_controls()
        for control in available_controls:
            value = self.settings_manager.get_control(control)
            if value is not None: settings[control] = value
        if settings: self.settings_queried.emit(settings)

    @pyqtSlot(dict)
    def apply_settings(self, settings_dict):
        """Applies a dictionary of new settings to the camera using the settings manager."""
        self.last_applied_settings = settings_dict

        print(f"Applying settings to camera {self.device_path}: {settings_dict}")
        if hasattr(self.settings_manager, 'set_all_controls'):
            batch_results = self.settings_manager.set_all_controls(settings_dict)
            if batch_results and all(batch_results.values()):
                print(f"✓ Batch applied all {len(settings_dict)} settings successfully")
                self.verify_settings_new(settings_dict)
                return
        
        for name, value in settings_dict.items():
            self.settings_manager.set_control(name, value)
        self.verify_settings_new(settings_dict)

    def verify_settings_new(self, expected_settings):
        """Verify that settings were actually applied correctly using the settings manager."""
        print(f"Verifying settings for camera {self.device_path}...")
        all_correct = True
        for name, expected_value in expected_settings.items():
            actual_value = self.settings_manager.get_control(name)
            if actual_value is not None and abs(actual_value - expected_value) <= 1:
                print(f"  ✓ {name}: expected={expected_value}, actual={actual_value}")
            else:
                print(f"  ✗ {name}: expected={expected_value}, actual={actual_value} [MISMATCH]")
                all_correct = False
        if all_correct: print("All settings verified successfully!")
        else: print("Some settings may not have been applied correctly.")
        return all_correct

    def gstreamer_pipeline(self, device, capture_width=1280, capture_height=720, framerate=30):
        """Constructs a GStreamer pipeline string for capturing from a V4L2 device."""
        return (
            f"v4l2src device={device} ! "
            f"video/x-raw, width=(int){capture_width}, height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
        )

    def try_initialize_camera(self):
        """Try different methods to initialize the camera, optimized for speed."""
        print(f"Initializing camera at {self.device_path}...")
        try:
            device_index = int(''.join(filter(str.isdigit, self.device_path)))
            self.cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
            if self.cap.isOpened():
                print("Camera opened successfully with V4L2 backend")
                self._using_gstreamer = False
                return True
            self.cap = cv2.VideoCapture(device_index)
            if self.cap.isOpened():
                print("Camera opened successfully by index")
                self._using_gstreamer = False
                return True
        except (ValueError, TypeError): pass
        
        self.cap = cv2.VideoCapture(self.device_path)
        if self.cap.isOpened():
            print("Camera opened successfully by path")
            self._using_gstreamer = False
            return True
        
        pipeline = self.gstreamer_pipeline(device=self.device_path)
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if self.cap.isOpened():
            print("Camera opened successfully with GStreamer")
            self._using_gstreamer = True
            return True
        
        print(f"Error: All methods failed. Could not open camera at {self.device_path}.")
        return False

    def run(self):
        """The main loop for capturing frames from the camera."""
        if not self.try_initialize_camera():
            self.camera_failed.emit()
            self.running = False
            return
        
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not hasattr(self, '_using_gstreamer') or not self._using_gstreamer:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                print(f"Requested 30 FPS from camera {self.device_path}")
        except Exception as e:
            print(f"Note: Some camera properties could not be set: {e}")
        
        self.load_and_apply_saved_settings()
        self.camera_ready.emit()
        
        print("Camera started successfully.")
        while self.running:
            ret, frame = self.cap.read()
            
            if ret and frame is not None:
                if self.failure_count > 0:
                    print("Camera stream recovered.")
                self.failure_count = 0
                
                # --- START: Apply crop before emitting ---
                cropped_frame = self._crop_frame(frame)
                self.ui_frame_ready.emit(cropped_frame)
                self.inference_frame_ready.emit(cropped_frame)
                # --- END: Apply crop ---
            else:
                self.failure_count += 1
                print(f"Warning: Failed to grab frame from {self.device_path}. Failure count: {self.failure_count}")
                
                if self.failure_count >= self.failure_threshold:
                    print(f"Error: Exceeded failure threshold. Attempting to re-initialize camera...")
                    if self.cap:
                        self.cap.release()
                    
                    if self.try_initialize_camera():
                        print("Camera re-initialized successfully.")
                        try:
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            if not hasattr(self, '_using_gstreamer') or not self._using_gstreamer:
                                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                self.cap.set(cv2.CAP_PROP_FPS, 30)
                        except Exception as e:
                            print(f"Note: Could not set basic properties after recovery: {e}")
                        
                        if self.last_applied_settings:
                            print(f"Re-applying last known settings: {self.last_applied_settings}")
                            self.apply_settings(self.last_applied_settings)
                        else:
                            self.load_and_apply_saved_settings()

                        self.failure_count = 0 
                    else:
                        print(f"Fatal: Could not re-initialize camera. Stopping worker.")
                        self.camera_failed.emit()
                        self.running = False
                
                self.msleep(100)

        if self.cap:
            self.cap.release()
        print(f"Camera {self.device_path} released.")

    def load_and_apply_saved_settings(self):
        """Reads the config.ini file and applies any saved settings for this device."""
        print(f"Loading and applying saved settings for {self.device_path}...")
        config = configparser.ConfigParser()
        if not os.path.exists('config.ini'): return

        config.read('config.ini')
        section_name = f"Settings:{self.device_path}"
        if config.has_section(section_name):
            saved_settings = {}
            for key, value in config.items(section_name):
                try: saved_settings[key] = int(value)
                except ValueError: pass
            
            if saved_settings:
                print(f"Applying saved settings to camera: {saved_settings}")
                self.msleep(200)
                self.last_applied_settings = saved_settings
                if self.apply_settings_with_retry(saved_settings):
                    print("Successfully applied all saved settings!")
                else:
                    print("Some settings may not have been applied correctly.")

    def apply_settings_with_retry(self, settings_dict, max_retries=2):
        """Apply settings with retry logic using the settings manager."""
        if hasattr(self.settings_manager, 'set_all_controls'):
            print(f"Attempting batch application of {len(settings_dict)} settings")
            batch_results = self.settings_manager.set_all_controls(settings_dict)
            if batch_results and all(batch_results.values()):
                print("✓ Batch application successful")
                return True
        
        for attempt in range(max_retries):
            all_success = True
            for name, value in settings_dict.items():
                try:
                    if not self.settings_manager.set_control(name, value):
                        all_success = False
                except Exception as e:
                    print(f"  ERROR: Exception setting {name} to {value}: {e}")
                    all_success = False
            
            if all_success: return True
            if attempt < max_retries - 1:
                print(f"Retrying settings application in 100ms...")
                self.msleep(100)
        
        return False

    def stop(self):
        """
        Sets the running flag to False and waits for the thread to finish.
        This is a blocking call to ensure safe shutdown.
        """
        print(f"Stopping camera worker for {self.device_path}...")
        self.running = False
        if not self.wait(2000): # Wait for up to 2 seconds
            print(f"Warning: Camera worker for {self.device_path} did not stop in time.")

