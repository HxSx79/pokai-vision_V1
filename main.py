import sys
import os
import glob
from pathlib import Path
import zmq
import time
import configparser
import cv2
import json
import numpy as np
from pyzbar.pyzbar import decode
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QComboBox, QFrame,
                             QGridLayout, QDialog, QDialogButtonBox, QListWidget,
                             QListWidgetItem, QSlider, QLineEdit, QCheckBox, QSplashScreen, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QIntValidator, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot, QMutex, QMutexLocker, QSize, QPoint, QRect

from camera_worker import CameraWorker

# --- START: New Crop Dialog Classes ---

class CropLabel(QLabel):
    """A custom QLabel to handle drawing a cropping rectangle."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.begin = QPoint()
        self.end = QPoint()
        self.drawing = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.begin = event.pos()
            self.end = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        rect = QRect(self.begin, self.end).normalized()
        if not rect.isNull():
            painter = QPainter(self)
            pen = QPen(Qt.red, 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(rect)

    def get_rect(self):
        """Returns the drawn rectangle."""
        return QRect(self.begin, self.end).normalized()

class CropDialog(QDialog):
    """A dialog to define a crop area on a camera frame."""
    crop_updated = pyqtSignal(object)  # Emits a tuple (x, y, w, h) or None

    def __init__(self, frame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Camera Crop")
        self.setMinimumSize(800, 600)

        self.original_frame = frame
        h, w, _ = self.original_frame.shape
        self.original_size = (w, h)

        main_layout = QVBoxLayout(self)

        self.crop_label = CropLabel()
        # --- FIX: Ensure NumPy array is contiguous for QImage constructor ---
        frame_contiguous = np.ascontiguousarray(self.original_frame)
        q_img = QImage(frame_contiguous.data, w, h, w * 3, QImage.Format_RGB888).rgbSwapped()
        self.pixmap = QPixmap.fromImage(q_img)
        self.crop_label.setPixmap(self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.crop_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.crop_label, 1)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Reset | QDialogButtonBox.Cancel)
        self.button_box.button(QDialogButtonBox.Save).clicked.connect(self.save_crop)
        self.button_box.button(QDialogButtonBox.Reset).clicked.connect(self.reset_crop)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)
    
    def resizeEvent(self, event):
        """Handle window resize to scale the pixmap."""
        self.crop_label.setPixmap(self.pixmap.scaled(self.crop_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def save_crop(self):
        drawn_rect = self.crop_label.get_rect()
        
        # Scale the drawn rectangle back to the original frame dimensions
        label_size = self.crop_label.size()
        pixmap = self.crop_label.pixmap()
        if pixmap is None or pixmap.isNull():
            return

        pixmap_size = pixmap.size()
        
        # Calculate the offset of the pixmap within the label (due to KeepAspectRatio)
        offset_x = (label_size.width() - pixmap_size.width()) / 2
        offset_y = (label_size.height() - pixmap_size.height()) / 2

        # Adjust drawn rectangle coordinates to be relative to the pixmap
        adjusted_x = drawn_rect.x() - offset_x
        adjusted_y = drawn_rect.y() - offset_y

        scale_x = self.original_size[0] / pixmap_size.width()
        scale_y = self.original_size[1] / pixmap_size.height()
        
        x = int(adjusted_x * scale_x)
        y = int(adjusted_y * scale_y)
        w = int(drawn_rect.width() * scale_x)
        h = int(drawn_rect.height() * scale_y)

        # Clamp values to be within the frame dimensions
        x = max(0, x)
        y = max(0, y)
        w = min(w, self.original_size[0] - x)
        h = min(h, self.original_size[1] - y)

        if w > 0 and h > 0:
            self.crop_updated.emit((x, y, w, h))
        self.accept()

    def reset_crop(self):
        self.crop_updated.emit(None) # Emit None to signify reset
        self.accept()

# --- END: New Crop Dialog Classes ---


class CameraAssignmentDialog(QDialog):
    # This class remains unchanged.
    assignments_changed = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Camera Roles")
        self.setMinimumSize(1024, 550)
        self.camera_workers = {}
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.available_devices = sorted(glob.glob("/dev/video*"))
        main_layout = QVBoxLayout(self)
        feeds_layout = QHBoxLayout()
        main_layout.addLayout(feeds_layout)
        self.main_cam_panel, self.main_cam_combo = self.create_assignment_panel("Main Camera")
        self.qr_cam_panel, self.qr_cam_combo = self.create_assignment_panel("QR Camera")
        feeds_layout.addWidget(self.main_cam_panel)
        feeds_layout.addWidget(self.qr_cam_panel)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)
    def create_assignment_panel(self, role_name):
        panel = QFrame(); panel.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(panel)
        title = QLabel(f"<b>{role_name}</b>"); title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        combo = QComboBox(); combo.addItems(["Select Device"] + self.available_devices)
        layout.addWidget(combo)
        video_label = QLabel("Select a device from the dropdown")
        video_label.setMinimumSize(480, 360); video_label.setAlignment(Qt.AlignCenter)
        video_label.setStyleSheet("background-color: black; color: white;")
        layout.addWidget(video_label)
        combo.currentTextChanged.connect(lambda device, r=role_name, lbl=video_label: self.on_device_selected(r, device, lbl))
        current_device = self.config.get('General', f"{role_name.lower().replace(' ', '_')}_device", fallback=None)
        if current_device in self.available_devices: combo.setCurrentText(current_device)
        else: self.on_device_selected(role_name, "Select Device", video_label)
        return panel, combo
    def on_device_selected(self, role_name, device_path, video_label):
        if role_name in self.camera_workers: self.camera_workers[role_name].stop()
        if device_path == "Select Device":
            video_label.setText("Select a device from the dropdown")
            video_label.setStyleSheet("background-color: black; color: white;")
            video_label.setPixmap(QPixmap())
            return
        video_label.setText(f"Loading {device_path}...")
        worker = CameraWorker(device_path)
        worker.ui_frame_ready.connect(lambda frame, lbl=video_label: self.update_feed(frame, lbl))
        worker.camera_failed.connect(lambda path=device_path, lbl=video_label: self.on_camera_failed(path, lbl))
        worker.finished.connect(worker.deleteLater)
        worker.start()
        self.camera_workers[role_name] = worker
    def on_camera_failed(self, device_path, label_widget):
        label_widget.setText(f"Failed to open\n{device_path}")
        label_widget.setStyleSheet("background-color: black; color: red; font-size: 18px; font-weight: bold;")
    def update_feed(self, frame, label_widget):
        h, w, ch = frame.shape
        q_img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
        label_widget.setPixmap(QPixmap.fromImage(q_img).scaled(label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    def save_assignments(self):
        main_cam_device = self.main_cam_combo.currentText()
        qr_cam_device = self.qr_cam_combo.currentText()
        if main_cam_device == "Select Device": main_cam_device = 'none'
        if qr_cam_device == "Select Device": qr_cam_device = 'none'
        if 'General' not in self.config: self.config.add_section('General')
        self.config['General']['main_camera_device'] = main_cam_device
        self.config['General']['qr_camera_device'] = qr_cam_device
        with open('config.ini', 'w') as configfile: self.config.write(configfile)
        self.assignments_changed.emit()
    def stop_all_workers(self):
        for worker in self.camera_workers.values():
            if worker and worker.isRunning(): worker.stop()
        self.camera_workers.clear()
    def accept(self): self.save_assignments(); self.stop_all_workers(); super().accept()
    def reject(self): self.stop_all_workers(); super().reject()

class CameraSettingsDialog(QDialog):
    """A dialog to adjust settings for a selected camera."""
    def __init__(self, main_cam, qr_cam, model_dir, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Settings")
        self.setMinimumWidth(600)
        self.device_paths = {"Main Camera": main_cam, "QR Camera": qr_cam}
        self.settings_manager = None
        self.current_device = None
        self.model_dir = model_dir
        self.main_window = parent

        layout = QVBoxLayout(self)
        top_layout = QHBoxLayout()
        assign_button = QPushButton("Assign Camera Roles...")
        assign_button.clicked.connect(self.open_assignment_dialog)
        top_layout.addWidget(assign_button)
        
        # --- START: Move crop button and style it ---
        crop_button = QPushButton("Set Camera Crop...")
        crop_button.clicked.connect(self.open_crop_dialog)
        top_layout.addWidget(crop_button)
        # --- END: Move crop button ---
        
        top_layout.addStretch()
        layout.addLayout(top_layout)

        selection_layout = QHBoxLayout()
        selection_layout.addWidget(QLabel("Configure Settings for:"))
        self.cam_select_combo = QComboBox()
        self.cam_select_combo.addItems(self.device_paths.keys())
        self.cam_select_combo.currentTextChanged.connect(self.on_camera_selected)
        selection_layout.addWidget(self.cam_select_combo)
        layout.addLayout(selection_layout)
        self.grid = QGridLayout()
        layout.addLayout(self.grid)
        self.sliders = {}; self.slider_labels = {}
        self.default_settings = {"brightness": 0, "contrast": 32, "saturation": 64, "gain": 0, "sharpness": 5, "white_balance_automatic": 1}
        self.slider_ranges = {"brightness": (-64, 64), "contrast": (0, 64), "saturation": (1, 128), "gain": (0, 100), "sharpness": (0, 6)}
        self.button_box = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        self.button_box.button(QDialogButtonBox.Apply).clicked.connect(self.on_apply)
        self.button_box.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.on_reset_to_default)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        self.on_camera_selected(self.cam_select_combo.currentText())

    def open_assignment_dialog(self):
        dialog = CameraAssignmentDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.parent().schedule_restart()
            self.accept() 
    
    def open_crop_dialog(self):
        """Opens the dialog to define the cropping area."""
        if not self.main_window or self.main_window.latest_raw_frame is None:
            print("[Error] No camera frame available to set crop.")
            return

        frame = self.main_window.latest_raw_frame
        dialog = CropDialog(frame, parent=self)
        dialog.crop_updated.connect(self.save_crop_settings)
        dialog.exec_()

    def save_crop_settings(self, crop_rect):
        """Saves the crop rectangle to the model settings file."""
        if not self.model_dir:
            print("[Error] Cannot save crop settings, no model loaded.")
            return

        settings_path = self.model_dir / 'model_settings.ini'
        config = configparser.ConfigParser()
        if settings_path.is_file():
            config.read(settings_path)

        if crop_rect:
            if not config.has_section('Crop'):
                config.add_section('Crop')
            x, y, w, h = crop_rect
            config.set('Crop', 'x', str(x))
            config.set('Crop', 'y', str(y))
            config.set('Crop', 'w', str(w))
            config.set('Crop', 'h', str(h))
            print(f"Saved new crop settings: {crop_rect}")
        else: # Reset
            if config.has_section('Crop'):
                config.remove_section('Crop')
            print("Reset crop settings.")
        
        with open(settings_path, 'w') as configfile:
            config.write(configfile)
        
        # Apply live
        if self.main_window.camera_worker:
            self.main_window.camera_worker.set_crop(crop_rect)

    def on_camera_selected(self, role_name):
        device_path = self.device_paths.get(role_name)
        if not device_path or device_path == 'none':
            self.clear_grid()
            self.grid.addWidget(QLabel(f"{role_name} is not assigned."), 0, 0)
            return
        self.current_device = device_path
        self.clear_grid()
        self.grid.addWidget(QLabel(f"Loading settings for {role_name}..."), 0, 0)
        QTimer.singleShot(100, lambda: self.load_camera_settings_only(device_path))
    def load_camera_settings_only(self, device_path):
        try:
            from simple_camera_settings_manager import SimpleCameraSettingsManager
            self.settings_manager = SimpleCameraSettingsManager(device_path)
            current_settings = self.settings_manager.get_all_controls()
            if not current_settings:
                self.clear_grid()
                self.grid.addWidget(QLabel("No camera controls available for this device."), 0, 0)
                return
            self.populate_sliders(current_settings)
        except Exception as e:
            self.clear_grid()
            self.grid.addWidget(QLabel(f"Error loading settings: {str(e)}"), 0, 0)
    @pyqtSlot(dict)
    def populate_sliders(self, current_settings):
        self.clear_grid()
        if not current_settings:
            self.grid.addWidget(QLabel("No adjustable settings found for this camera."), 0, 0)
            return
        row = 0
        for name, value in sorted(current_settings.items()):
            if name == "white_balance_automatic":
                self.awb_checkbox = QCheckBox("Auto White Balance")
                self.awb_checkbox.setChecked(bool(value))
                self.grid.addWidget(self.awb_checkbox, row, 1)
            else:
                min_val, max_val = self.slider_ranges.get(name, (0, 100))
                slider, label = self.create_slider(name, int(value), min_val, max_val)
                self.sliders[name] = slider
                self.slider_labels[name] = label
                self.grid.addWidget(QLabel(f"{name.replace('_', ' ').title()}:"), row, 0)
                self.grid.addWidget(slider, row, 1)
                self.grid.addWidget(label, row, 2)
            row += 1
    def create_slider(self, name, current_val, min_val, max_val):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(current_val)
        label = QLabel(str(current_val))
        label.setFixedWidth(40)
        slider.valueChanged.connect(lambda v, l=label: l.setText(str(v)))
        return slider, label
    def clear_grid(self):
        for i in reversed(range(self.grid.count())): 
            widget = self.grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.sliders.clear(); self.slider_labels.clear()
    def on_apply(self):
        new_settings = {}
        for name, slider in self.sliders.items():
            new_settings[name] = slider.value()
        if hasattr(self, 'awb_checkbox'):
            new_settings["white_balance_automatic"] = int(self.awb_checkbox.isChecked())
        if not new_settings: return
        
        if self.settings_manager:
            self.settings_manager.set_all_controls(new_settings)
        
        self.save_settings_to_model_config(new_settings)
        
        if hasattr(self.parent(), 'camera_worker') and self.parent().camera_worker:
            if self.parent().config.get('General', 'main_camera_device', fallback='none') == self.current_device:
                self.parent().camera_worker.apply_settings(new_settings)
        
        self.accept()

    def save_settings_to_model_config(self, settings_to_save):
        if not self.model_dir:
            print("[Error] Cannot save camera settings, no model loaded.")
            return False
            
        settings_path = self.model_dir / 'model_settings.ini'
        print(f"Saving camera settings to {settings_path}...")
        
        try:
            config = configparser.ConfigParser()
            if settings_path.is_file():
                config.read(settings_path)

            if not config.has_section('Camera'):
                config.add_section('Camera')
            
            for key, value in settings_to_save.items():
                config.set('Camera', key, str(value))
            
            with open(settings_path, 'w') as configfile:
                config.write(configfile)
            
            print("Camera settings saved successfully.")
            return True
        except Exception as e:
            print(f"Error saving camera settings to {settings_path}: {e}")
            return False

    def on_reset_to_default(self):
        for name, slider in self.sliders.items():
            slider.setValue(self.default_settings.get(name, 0))
        if hasattr(self, 'awb_checkbox'):
            self.awb_checkbox.setChecked(bool(self.default_settings.get("white_balance_automatic", True)))
    def reject(self):
        super().reject()

class OperatorDialog(QDialog):
    # This class remains unchanged.
    operator_validated = pyqtSignal(dict)
    def __init__(self, device_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Operator Change - Scan QR Code")
        self.setMinimumSize(800, 600)
        self.main_layout = QVBoxLayout(self)
        self.video_label = QLabel("Initializing QR Camera...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white; font-size: 18px;")
        self.main_layout.addWidget(self.video_label, 1)
        self.confirmation_widget = QWidget()
        self.confirmation_layout = QGridLayout(self.confirmation_widget)
        self.op_info_labels = {"Company": QLabel("N/A"), "First Name": QLabel("N/A"), "Last Name": QLabel("N/A"), "Position": QLabel("N/A")}
        row = 0
        for title, label_widget in self.op_info_labels.items():
            title_label = QLabel(f"{title}:"); title_label.setFont(QFont("Arial", 14, QFont.Bold))
            label_widget.setFont(QFont("Arial", 14)); self.confirmation_layout.addWidget(title_label, row, 0); self.confirmation_layout.addWidget(label_widget, row, 1); row += 1
        self.main_layout.addWidget(self.confirmation_widget); self.confirmation_widget.hide()
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.button(QDialogButtonBox.Ok).setText("Validate"); self.button_box.accepted.connect(self.on_validate); self.button_box.rejected.connect(self.reject)
        self.main_layout.addWidget(self.button_box); self.button_box.hide()
        self.camera_worker = CameraWorker(device_path)
        self.camera_worker.ui_frame_ready.connect(self.process_qr_frame)
        self.camera_worker.start()
        self.operator_data = None
    @pyqtSlot(object)
    def process_qr_frame(self, frame):
        if self.confirmation_widget.isVisible(): return
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        decoded_objects = decode(gray_frame)
        if decoded_objects:
            if self.camera_worker: self.camera_worker.stop() 
            self.parse_and_confirm(decoded_objects[0].data.decode('utf-8'))
            return
        h, w, ch = frame.shape
        q_img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    def parse_and_confirm(self, qr_data):
        try:
            parts = [p.strip() for p in qr_data.split('/')]
            if len(parts) == 4:
                self.operator_data = {"Company": parts[0], "First Name": parts[1], "Last Name": parts[2], "Position": parts[3]}
                for key, value in self.operator_data.items():
                    if key in self.op_info_labels: self.op_info_labels[key].setText(value)
                self.video_label.hide(); self.confirmation_widget.show(); self.button_box.show(); self.setWindowTitle("Confirm Operator Details")
            else: self.video_label.setText(f"Invalid QR Code Format!\nRetrying..."); QTimer.singleShot(3000, self.restart_scanning)
        except Exception: self.video_label.setText("Error reading QR code.\nRetrying..."); QTimer.singleShot(3000, self.restart_scanning)
    def restart_scanning(self):
        self.confirmation_widget.hide()
        self.video_label.show()
        if self.camera_worker and not self.camera_worker.isRunning():
            self.video_label.setText("Initializing QR Camera...")
            self.camera_worker.start()
    def on_validate(self):
        if self.operator_data: 
            self.operator_validated.emit(self.operator_data)
            self.accept()
    def _cleanup_worker(self):
        if hasattr(self, 'camera_worker') and self.camera_worker is not None:
            if self.camera_worker.isRunning(): self.camera_worker.stop()
            self.camera_worker.deleteLater()
            self.camera_worker = None
    def accept(self): self._cleanup_worker(); super().accept()
    def reject(self): self._cleanup_worker(); super().reject()

class ConfidenceDialog(QDialog):
    confidence_changed = pyqtSignal(int)
    def __init__(self, current_confidence, model_dir, parent=None):
        super().__init__(parent)
        self.setWindowTitle("App Settings - Confidence Threshold")
        self.setMinimumWidth(400)
        self.model_dir = model_dir
        
        layout = QVBoxLayout(self)
        self.value_label = QLabel(f"{current_confidence}%")
        self.value_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.value_label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(current_confidence)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        layout.addWidget(self.slider)
        input_layout = QHBoxLayout()
        input_layout.addStretch()
        self.input_box = QLineEdit(str(current_confidence))
        self.input_box.setValidator(QIntValidator(0, 100))
        self.input_box.setFixedWidth(50)
        self.input_box.setAlignment(Qt.AlignCenter)
        input_layout.addWidget(self.input_box)
        input_layout.addWidget(QLabel("%"))
        input_layout.addStretch()
        layout.addLayout(input_layout)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.on_accepted)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        self.slider.valueChanged.connect(self.slider_changed)
        self.input_box.textChanged.connect(self.input_changed)
    def slider_changed(self, value):
        self.value_label.setText(f"{value}%")
        self.input_box.setText(str(value))
    def input_changed(self, text):
        if text and text.isdigit():
            value = int(text)
            if 0 <= value <= 100:
                self.value_label.setText(f"{value}%")
                self.slider.setValue(value)
    def on_accepted(self):
        new_confidence = self.slider.value()
        self.save_confidence_to_model_config(new_confidence)
        self.confidence_changed.emit(new_confidence)
        self.accept()

    def save_confidence_to_model_config(self, confidence_value):
        if not self.model_dir:
            print("[Error] Cannot save confidence, no model loaded.")
            return False
            
        settings_path = self.model_dir / 'model_settings.ini'
        print(f"Saving confidence to {settings_path}...")
        
        try:
            config = configparser.ConfigParser()
            if settings_path.is_file():
                config.read(settings_path)

            if not config.has_section('Confidence'):
                config.add_section('Confidence')
            
            config.set('Confidence', 'threshold', str(confidence_value))
            
            with open(settings_path, 'w') as configfile:
                config.write(configfile)
            
            print("Confidence setting saved successfully.")
            return True
        except Exception as e:
            print(f"Error saving confidence to {settings_path}: {e}")
            return False

class SelectionDialog(QDialog):
    # This class remains unchanged.
    model_selected = pyqtSignal(str, str, str, tuple, str)
    def __init__(self, model_base_dir, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Model")
        self.model_base_dir = Path(model_base_dir)
        self.setMinimumWidth(400)
        layout = QVBoxLayout(self)
        form_layout = QGridLayout()
        form_layout.addWidget(QLabel("Customer:"), 0, 0)
        self.customer_dropdown = QComboBox()
        form_layout.addWidget(self.customer_dropdown, 0, 1)
        form_layout.addWidget(QLabel("Program:"), 1, 0)
        self.program_dropdown = QComboBox()
        form_layout.addWidget(self.program_dropdown, 1, 1)
        form_layout.addWidget(QLabel("Part Number:"), 2, 0)
        self.part_number_dropdown = QComboBox()
        form_layout.addWidget(self.part_number_dropdown, 2, 1)
        layout.addLayout(form_layout)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.on_accepted)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        self.customer_dropdown.currentTextChanged.connect(self.populate_programs)
        self.program_dropdown.currentTextChanged.connect(self.populate_part_numbers)
        self.populate_customers()
    def populate_customers(self):
        self.customer_dropdown.clear()
        self.customer_dropdown.addItem("Select...")
        if self.model_base_dir.exists():
            for customer_dir in sorted(self.model_base_dir.iterdir()):
                if customer_dir.is_dir(): self.customer_dropdown.addItem(customer_dir.name)
    def populate_programs(self, customer_name):
        self.program_dropdown.clear(); self.part_number_dropdown.clear()
        self.program_dropdown.addItem("Select...")
        if customer_name and customer_name != "Select...":
            customer_path = self.model_base_dir / customer_name
            if customer_path.exists():
                for program_dir in sorted(customer_path.iterdir()):
                    if program_dir.is_dir(): self.program_dropdown.addItem(program_dir.name)
    def populate_part_numbers(self, program_name):
        self.part_number_dropdown.clear()
        self.part_number_dropdown.addItem("Select...")
        customer_name = self.customer_dropdown.currentText()
        if program_name and program_name != "Select...":
            program_path = self.model_base_dir / customer_name / program_name
            if program_path.exists():
                for part_dir in sorted(program_path.iterdir()):
                    if part_dir.is_dir(): self.part_number_dropdown.addItem(part_dir.name)
    def on_accepted(self):
        customer = self.customer_dropdown.currentText()
        program = self.program_dropdown.currentText()
        part_number = self.part_number_dropdown.currentText()
        if "Select..." in [customer, program, part_number]: return
        part_path = self.model_base_dir / customer / program / part_number
        model_path = part_path / "best.pt"
        logo_path = self.model_base_dir / customer / "customer-logo.png"
        cad_path = part_path / f"{part_number}.png"
        id_card_path = part_path / "id.card"
        if not model_path.exists(): return
        part_description = "N/A"
        if id_card_path.exists():
            try: part_description = id_card_path.read_text().strip()
            except Exception as e: print(f"Error reading id.card: {e}")
        info_tuple = (customer, program, part_number)
        self.model_selected.emit(str(model_path), str(logo_path), str(cad_path), info_tuple, part_description)
        self.accept()

class PokaiVisionUI(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pokai Vision")
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        self.config = configparser.ConfigParser()
        self.zmq_context = zmq.Context()
        self.camera_worker = None
        self.zmq_thread = None
        self.oldPos = None
        
        self.current_model_dir = None
        self.latest_raw_frame = None
        self.last_known_detections = []
        self.last_inference_ms = 0.0
        self.display_fps = 0.0
        self._display_fps_time = time.time()
        self._display_frame_count = 0
        self.display_update_timer = QTimer(self)
        self.display_update_timer.timeout.connect(self.update_display)
        # --- START: Add flag for detection status ---
        self.first_detection_received = False
        # --- END: Add flag ---

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.setup_ui()
        QTimer.singleShot(100, self.load_config_and_start_services)

    def load_config_and_start_services(self):
        """Initial setup of services."""
        print("Loading configuration and starting services...")
        
        if not os.path.exists('config.ini'):
            print("config.ini not found, creating default config...")
            self.create_default_config()
        
        self.config.read('config.ini')
        self.current_confidence = self.config.getint('General', 'default_confidence', fallback=80)
        
        print(f"Loaded config with confidence: {self.current_confidence}")
        
        self.start_camera_feed()
        self.start_zmq_client()
        self.display_update_timer.start(33) # ~30 FPS

    def create_default_config(self):
        config = configparser.ConfigParser()
        config.add_section('General')
        config.set('General', 'main_camera_device', 'none')
        config.set('General', 'qr_camera_device', 'none')
        config.set('General', 'default_confidence', '80')
        config.set('General', 'model_base_dir', 'yolo_models/Customers')
        config.add_section('ZMQ')
        config.set('ZMQ', 'server_address', 'tcp://localhost:5555')
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
        print("Default config.ini created")

    def schedule_restart(self):
        print("Scheduling a restart of services...")
        self.display_update_timer.stop()
        QTimer.singleShot(10, self.reload_config_and_restart)

    def reload_config_and_restart(self):
        print("Performing service restart...")
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker = None
        if self.zmq_thread:
            self.zmq_thread.stop()
            self.zmq_thread = None
        self.config.read('config.ini')
        QTimer.singleShot(500, self.restart_services)
    
    def restart_services(self):
        print("Restarting services after config reload...")
        self.start_camera_feed()
        self.start_zmq_client()
        self.display_update_timer.start(33)

    def setup_ui(self):
        if self.central_widget.layout() is not None:
            while self.central_widget.layout().count():
                child = self.central_widget.layout().takeAt(0)
                if child.widget(): child.widget().deleteLater()
        self.central_widget.setStyleSheet("background-color: #333333;")
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.title_bar = self.create_title_bar()
        main_layout.addWidget(self.title_bar)
        content_area = QWidget()
        content_area.setStyleSheet("background-color: #E0E0E0;")
        grid_layout = QGridLayout(content_area)
        grid_layout.setSpacing(10)
        grid_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addWidget(content_area, 1)
        q1 = self.create_info_panel()
        q2 = self.create_buttons_panel()
        q3 = self.create_latest_detections_panel()
        q4 = self.create_placeholder_panel("Quadrant 4")
        q5 = self.create_status_panel("PART DETECTION", "part")
        q6 = self.create_status_panel("CLIP DETECTION", "clip")
        q7 = self.create_camera_panel()
        q8 = self.create_cad_panel()
        grid_layout.addWidget(q1, 0, 0, 1, 2)
        grid_layout.addWidget(q2, 0, 2)
        grid_layout.addWidget(q3, 1, 0)
        grid_layout.addWidget(q5, 1, 1)
        grid_layout.addWidget(q7, 1, 2)
        grid_layout.addWidget(q4, 2, 0)
        grid_layout.addWidget(q6, 2, 1)
        grid_layout.addWidget(q8, 2, 2)
        grid_layout.setRowStretch(0, 12)
        grid_layout.setRowStretch(1, 44)
        grid_layout.setRowStretch(2, 44)
        grid_layout.setColumnStretch(0, 20)
        grid_layout.setColumnStretch(1, 40)
        grid_layout.setColumnStretch(2, 40)

    def create_title_bar(self):
        title_bar_widget = QFrame()
        title_bar_widget.setStyleSheet("background-color: #3C3C3C; color: white;")
        title_bar_widget.setFixedHeight(40)
        layout = QHBoxLayout(title_bar_widget)
        layout.setContentsMargins(15, 0, 5, 0)
        title_label = QLabel("pok.AI - vision")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.datetime_label = QLabel("...")
        self.datetime_label.setFont(QFont("Arial", 12))
        self.datetime_label.setAlignment(Qt.AlignCenter)
        self.operator_label = QLabel("Operator: N/A")
        self.operator_label.setFont(QFont("Arial", 12))
        self.operator_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        button_style = "QPushButton { background-color: #E0E0E0; color: #333333; border: none; border-radius: 14px; font-family: 'Ubuntu'; font-weight: bold; font-size: 16px; } QPushButton:hover { background-color: #D0D0D0; }"
        minimize_button = QPushButton("—")
        minimize_button.setFixedSize(28, 28)
        minimize_button.setStyleSheet(button_style)
        minimize_button.clicked.connect(self.showMinimized)
        close_button = QPushButton("✕")
        close_button.setFixedSize(28, 28)
        close_button.setStyleSheet(button_style + "QPushButton:hover { background-color: #E81123; color: white; }")
        close_button.clicked.connect(self.close)
        layout.addWidget(title_label)
        layout.addStretch()
        layout.addWidget(self.datetime_label)
        layout.addStretch()
        layout.addWidget(self.operator_label)
        layout.addSpacing(10)
        layout.addWidget(minimize_button)
        layout.addWidget(close_button)
        self.datetime_timer = QTimer(self)
        self.datetime_timer.timeout.connect(self.update_datetime)
        self.datetime_timer.start(1000)
        return title_bar_widget
    def update_datetime(self):
        from datetime import datetime
        self.datetime_label.setText(datetime.now().strftime("%A, %B %d, %Y %H:%M:%S"))
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton: self.oldPos = event.globalPos()
    def mouseMoveEvent(self, event):
        if self.oldPos:
            delta = QPoint(event.globalPos() - self.oldPos)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.oldPos = event.globalPos()
    def mouseReleaseEvent(self, event): self.oldPos = None
    def create_info_panel(self):
        frame = self.create_styled_frame()
        layout = QHBoxLayout(frame)
        logo_frame = self.create_styled_frame()
        logo_layout = QVBoxLayout(logo_frame)
        self.logo_label = QLabel()
        self.logo_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.logo_label.setAlignment(Qt.AlignCenter)
        pk_icon_path = "icons/pk_icon.png"
        if os.path.exists(pk_icon_path):
            self.logo_label.setPixmap(QPixmap(pk_icon_path).scaled(200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else: self.logo_label.setText("pok.AI")
        logo_layout.addWidget(self.logo_label)
        layout.addWidget(logo_frame, 1)
        layout.addSpacing(50)
        info_grid_widget = QWidget()
        info_grid_layout = QGridLayout(info_grid_widget)
        info_grid_layout.setSpacing(5)
        customer_frame, self.info_customer_value_label = self.create_info_box("CUSTOMER")
        info_grid_layout.addWidget(customer_frame, 0, 0)
        program_frame, self.info_program_value_label = self.create_info_box("PROGRAM")
        info_grid_layout.addWidget(program_frame, 0, 1)
        part_number_frame, self.info_part_number_value_label = self.create_info_box("PART NUMBER")
        info_grid_layout.addWidget(part_number_frame, 1, 0)
        description_frame, self.info_description_value_label = self.create_info_box("DESCRIPTION")
        info_grid_layout.addWidget(description_frame, 1, 1)
        layout.addWidget(info_grid_widget, 2)
        return frame
    
    def create_buttons_panel(self):
        frame = self.create_styled_frame()
        layout = QHBoxLayout(frame)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignCenter)
        b1 = self.create_icon_button("icons/change_model.png", "Change\nPart")
        b2 = self.create_icon_button("icons/camera_setup.png", "Camera\nSettings")
        b3 = self.create_icon_button("icons/app_settings.png", "App\nSettings")
        b4 = self.create_icon_button("icons/op_change.png", "Operator\nChange")
        b5 = self.create_icon_button("icons/quality_alert.png", "Quality\nAlert")

        b1.findChild(QPushButton).clicked.connect(self.open_selection_dialog)
        b2.findChild(QPushButton).clicked.connect(self.open_camera_settings_dialog)
        b3.findChild(QPushButton).clicked.connect(self.open_confidence_dialog)
        b4.findChild(QPushButton).clicked.connect(self.open_operator_dialog)
        b5.findChild(QPushButton).clicked.connect(self.on_quality_alert_clicked)

        layout.addStretch()
        layout.addWidget(b1)
        layout.addStretch()
        layout.addWidget(b2)
        layout.addStretch()
        layout.addWidget(b3)
        layout.addStretch()
        layout.addWidget(b4)
        layout.addStretch()
        layout.addWidget(b5)
        layout.addStretch()
        return frame

    def create_icon_button(self, icon_path, text):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0,0,0,0); layout.setSpacing(5)
        layout.setAlignment(Qt.AlignCenter)
        button = QPushButton()
        if os.path.exists(icon_path): button.setIcon(QIcon(icon_path))
        button.setIconSize(QSize(64, 64)); button.setFixedSize(80, 80)
        button.setStyleSheet("QPushButton { border: 2px solid #AAAAAA; border-radius: 5px; background-color: #F0F0F0; } QPushButton:hover { background-color: #D5D5D5; }")
        button.setCursor(Qt.PointingHandCursor)
        label = QLabel(text)
        label.setWordWrap(True); label.setAlignment(Qt.AlignHCenter)
        label.setFont(QFont("Arial", 10))
        layout.addWidget(button); layout.addWidget(label)
        return widget
    def create_latest_detections_panel(self):
        frame = self.create_styled_frame()
        layout = QVBoxLayout(frame)
        layout.addWidget(QLabel("Latest Detections"))
        self.history_list_widget = QListWidget()
        layout.addWidget(self.history_list_widget, 1)
        totals_label = QLabel("Totals placeholder...")
        totals_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(totals_label)
        return frame
    def create_placeholder_panel(self, text=""):
        frame = self.create_styled_frame()
        layout = QVBoxLayout(frame)
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        return frame
    def create_status_panel(self, text, indicator_name):
        outer_frame = self.create_styled_frame()
        outer_layout = QVBoxLayout(outer_frame)
        outer_layout.setContentsMargins(10, 10, 10, 10)
        indicator_label = QLabel(text)
        indicator_label.setAlignment(Qt.AlignCenter)
        indicator_label.setFont(QFont("Arial", 24, QFont.Bold))
        indicator_label.setStyleSheet("background-color: #B0B0B0; color: white; border: 2px solid gray; border-radius: 10px;")
        if indicator_name == "part": self.part_status_indicator = indicator_label
        elif indicator_name == "clip": self.clip_status_indicator = indicator_label
        outer_layout.addWidget(indicator_label)
        return outer_frame
    def create_camera_panel(self):
        frame = self.create_styled_frame()
        layout = QVBoxLayout(frame)
        layout.addWidget(QLabel("Live Camera Feed"), 0, Qt.AlignTop)
        self.video_label = QLabel("Initializing Camera...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label, 1)
        return frame
    def create_cad_panel(self):
        frame = self.create_styled_frame()
        layout = QVBoxLayout(frame)
        layout.addWidget(QLabel("CAD Picture"), 0, Qt.AlignTop)
        self.cad_label = QLabel("No Model Loaded")
        self.cad_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.cad_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.cad_label, 1)
        return frame
    def create_info_box(self, title_text):
        frame = self.create_styled_frame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(5,5,5,5)
        title_label = self.create_title_label(title_text)
        value_label = self.create_value_label("N/A")
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addStretch()
        return frame, value_label
    def create_styled_frame(self):
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Raised)
        frame.setStyleSheet("background-color: white; border-radius: 5px;")
        return frame
    def create_title_label(self, text):
        label = QLabel(text)
        label.setFont(QFont("Arial", 10, QFont.Bold))
        label.setStyleSheet("color: #555555; background-color: transparent;")
        return label
    def create_value_label(self, text):
        label = QLabel(text)
        label.setFont(QFont("Arial", 14))
        label.setStyleSheet("background-color: transparent;")
        return label

    def open_selection_dialog(self):
        model_dir = self.config.get('General', 'model_base_dir', fallback='yolo_models/Customers')
        dialog = SelectionDialog(model_base_dir=model_dir, parent=self)
        dialog.model_selected.connect(self.on_model_selected)
        dialog.exec_()
    def open_confidence_dialog(self):
        if not self.current_model_dir:
            print("[Warning] Cannot open confidence settings: No model loaded.")
            return
        dialog = ConfidenceDialog(self.current_confidence, self.current_model_dir, self)
        dialog.confidence_changed.connect(self.on_confidence_changed)
        dialog.exec_()
    def open_camera_settings_dialog(self):
        if not self.current_model_dir:
            print("[Warning] Cannot open camera settings: No model loaded.")
            return
        self.pause_services()
        main_cam = self.config.get('General', 'main_camera_device', fallback='none')
        qr_cam = self.config.get('General', 'qr_camera_device', fallback='none')
        dialog = CameraSettingsDialog(main_cam, qr_cam, self.current_model_dir, self)
        dialog.exec_()
        self.schedule_restart()
    def open_operator_dialog(self):
        self.pause_services()
        qr_cam_device = self.config.get('General', 'qr_camera_device', fallback='none')
        if qr_cam_device == 'none' or not os.path.exists(qr_cam_device):
            self.schedule_restart()
            return
        dialog = OperatorDialog(device_path=qr_cam_device, parent=self)
        dialog.operator_validated.connect(self.on_operator_validated)
        dialog.exec_()
        self.schedule_restart()

    def on_quality_alert_clicked(self):
        """Placeholder function for the Quality Alert button."""
        print("Quality Alert button clicked!")

    def pause_services(self):
        """Pauses the camera and ZMQ thread."""
        print("Pausing services...")
        self.display_update_timer.stop()
        if self.camera_worker and self.camera_worker.isRunning():
            self.camera_worker.stop()
        if self.zmq_thread and self.zmq_thread.isRunning():
            self.zmq_thread.pause()
        self.video_label.setText("Camera Paused")

    @pyqtSlot(dict)
    def on_operator_validated(self, op_data):
        first = op_data.get("First Name", ""); last = op_data.get("Last Name", "")
        operator_name = f"{first} {last}".strip()
        self.operator_label.setText(f"Operator: {operator_name}" if operator_name else "Operator: N/A")
        print(f"Operator changed to: {op_data}")

    @pyqtSlot(int)
    def on_confidence_changed(self, new_confidence):
        self.current_confidence = new_confidence
        print(f"Confidence updated to: {new_confidence}%")
        if self.zmq_thread:
            self.zmq_thread.set_confidence_threshold(new_confidence / 100.0)

    @pyqtSlot(str, str, str, tuple, str)
    def on_model_selected(self, model_path, logo_path, cad_path, info_tuple, part_description):
        self.reset_status_indicators()
        self.first_detection_received = False # Reset flag
        self.current_model_dir = Path(model_path).parent
        self.last_known_detections = [] 
        
        customer, program, part_number = info_tuple
        self.info_customer_value_label.setText(customer)
        self.info_program_value_label.setText(program)
        self.info_part_number_value_label.setText(part_description)
        
        if Path(logo_path).exists():
            self.logo_label.setPixmap(QPixmap(logo_path).scaled(self.logo_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else: self.logo_label.setText("Logo Not Found")
        
        if Path(cad_path).exists():
            self.cad_label.setPixmap(QPixmap(cad_path).scaled(self.cad_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else: self.cad_label.setText("CAD Image Not Found")
        
        if self.zmq_thread: self.zmq_thread.set_model_path(model_path)
        
        self.load_and_apply_model_settings()

    def load_and_apply_model_settings(self):
        """Loads settings from the current model's ini file and applies them."""
        if not self.current_model_dir: return

        settings_path = self.current_model_dir / 'model_settings.ini'
        
        # Load Camera Settings first
        if self.camera_worker:
            cam_settings = {}
            crop_rect = None
            if settings_path.is_file():
                model_config = configparser.ConfigParser()
                model_config.read(settings_path)
                if model_config.has_section('Camera'):
                    cam_settings = {key: int(value) for key, value in model_config.items('Camera')}
                if model_config.has_section('Crop'):
                    crop_rect = (
                        model_config.getint('Crop', 'x'),
                        model_config.getint('Crop', 'y'),
                        model_config.getint('Crop', 'w'),
                        model_config.getint('Crop', 'h')
                    )
            if cam_settings:
                print(f"Applying camera settings from model file: {cam_settings}")
                self.camera_worker.apply_settings(cam_settings)
            self.camera_worker.set_crop(crop_rect)

        # Then load confidence
        if settings_path.is_file():
            model_config = configparser.ConfigParser()
            model_config.read(settings_path)
            conf = model_config.getint('Confidence', 'threshold', fallback=self.current_confidence)
            self.on_confidence_changed(conf)
        else:
            print(f"No model_settings.ini found. Using current/default settings.")
            default_confidence = self.config.getint('General', 'default_confidence', fallback=80)
            self.on_confidence_changed(default_confidence)

    def reset_status_indicators(self):
        """Resets the detection status indicators to their initial gray state."""
        gray_style = "background-color: #B0B0B0; color: white; border: 2px solid gray; border-radius: 10px;"
        if hasattr(self, 'part_status_indicator'):
            self.part_status_indicator.setStyleSheet(gray_style)
        if hasattr(self, 'clip_status_indicator'):
            self.clip_status_indicator.setStyleSheet(gray_style)

    @pyqtSlot(str)
    def update_history(self, detection_result):
        self.history_list_widget.insertItem(0, detection_result)
        if self.history_list_widget.count() > 100:
            self.history_list_widget.takeItem(100)

    @pyqtSlot(str, str)
    def update_detection_statuses(self, part_status, clip_status):
        # --- START: Check flag before updating colors ---
        if not self.first_detection_received:
            return
        # --- END: Check flag ---
        part_color = "#90EE90" if part_status == 'PASS' else "#FF4C4C"
        clip_color = "#90EE90" if clip_status == 'PASS' else "#FF4C4C"
        part_style = f"background-color: {part_color}; color: white; border: 2px solid gray; border-radius: 10px;"
        clip_style = f"background-color: {clip_color}; color: white; border: 2px solid gray; border-radius: 10px;"
        self.part_status_indicator.setStyleSheet(part_style)
        self.clip_status_indicator.setStyleSheet(clip_style)

    def start_camera_feed(self):
        main_cam_device = self.config.get('General', 'main_camera_device', fallback='none')
        if main_cam_device == 'none' or not os.path.exists(main_cam_device):
            self.video_label.setText("No Main Camera assigned.\nPlease go to Camera Settings.")
            return
        
        print(f"Starting main camera feed for device: {main_cam_device}")
        self.camera_worker = CameraWorker(device_path=main_cam_device)
        self.camera_worker.ui_frame_ready.connect(self.handle_raw_frame)
        self.camera_worker.inference_frame_ready.connect(self.send_frame_to_zmq)
        self.camera_worker.camera_failed.connect(self.on_main_camera_failed)
        self.camera_worker.camera_ready.connect(self.on_main_camera_ready)
        self.camera_worker.start()

    def on_main_camera_ready(self):
        print("Main camera is ready.")
    
    def on_main_camera_failed(self):
        print("Main camera failed!")
        self.video_label.setText("Main Camera Failed!\nPlease check Camera Settings.")
        self.video_label.setStyleSheet("background-color: black; color: red; font-size: 18px;")

    def start_zmq_client(self):
        server_addr = self.config.get('ZMQ', 'server_address', fallback='tcp://localhost:5555')
        self.zmq_thread = ZmqClientThread(self.zmq_context, server_address=server_addr)
        self.zmq_thread.detections_ready.connect(self.handle_new_detections)
        self.zmq_thread.detection_result.connect(self.update_history)
        self.zmq_thread.detection_statuses.connect(self.update_detection_statuses)
        self.zmq_thread.start()

    @pyqtSlot(object)
    def send_frame_to_zmq(self, frame):
        if frame is not None and self.zmq_thread:
            self.zmq_thread.set_frame(frame)

    @pyqtSlot(object)
    def handle_raw_frame(self, frame):
        """Stores the latest raw frame from the camera."""
        self.latest_raw_frame = frame

    @pyqtSlot(list, float)
    def handle_new_detections(self, detections, inference_ms):
        """Stores the latest detection data from the ZMQ thread."""
        self.first_detection_received = True # Set flag on first result
        self.last_known_detections = detections
        self.last_inference_ms = inference_ms

    def update_display(self):
        """Draws last known detections on the latest camera frame."""
        if self.latest_raw_frame is None: return

        display_frame = self.latest_raw_frame.copy()

        if self.last_known_detections:
            box_color = (0, 255, 255); box_thickness = 2
            font_color = (0, 0, 0); font_scale = 0.6; font_thickness = 1
            for det in self.last_known_detections:
                box = det['box']; label = det['label']
                x1, y1, x2, y2 = box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, box_thickness)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                cv2.rectangle(display_frame, (x1, y1 - h - 5), (x1 + w, y1), box_color, -1)
                cv2.putText(display_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)

        self._display_frame_count += 1
        current_time = time.time()
        if (current_time - self._display_fps_time) >= 1.0:
            self.display_fps = self._display_frame_count / (current_time - self._display_fps_time)
            self._display_fps_time = current_time
            self._display_frame_count = 0
        
        overlay_text = f"Display FPS: {self.display_fps:.1f} | Inference: {self.last_inference_ms:.1f} ms"
        h, w, _ = display_frame.shape
        text_size, _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_x = w - text_size[0] - 10
        text_y = h - 10
        cv2.putText(display_frame, overlay_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        q_img = QImage(display_frame.data, w, h, w * 3, QImage.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event):
        print("Closing application...")
        self.display_update_timer.stop()
        if self.camera_worker: self.camera_worker.stop()
        if self.zmq_thread: self.zmq_thread.stop()
        event.accept()

class ZmqClientThread(QThread):
    detections_ready = pyqtSignal(list, float)
    detection_result = pyqtSignal(str)
    detection_statuses = pyqtSignal(str, str)
    
    def __init__(self, context, server_address, parent=None):
        super().__init__(parent)
        self.context = context
        self.server_address = server_address
        self.socket = None
        self.mutex = QMutex()
        self.running = True
        self.paused = False
        self.model_path_to_load = None
        self.confidence_to_set = None
        self.current_frame = None
        self.frame_pending = False

    def set_model_path(self, path):
        with QMutexLocker(self.mutex): self.model_path_to_load = path
    def set_frame(self, frame):
        if not self.frame_pending:
            with QMutexLocker(self.mutex):
                self.current_frame = frame
                self.frame_pending = True
    def set_confidence_threshold(self, value):
        with QMutexLocker(self.mutex): self.confidence_to_set = value
    def stop(self):
        with QMutexLocker(self.mutex): self.running = False
        self.wait()
    def pause(self):
        with QMutexLocker(self.mutex): self.paused = True
    def resume(self):
        with QMutexLocker(self.mutex): self.paused = False
    def create_and_connect_socket(self):
        if self.socket: self.socket.close()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.server_address)

    def run(self):
        from datetime import datetime
        self.create_and_connect_socket()
        
        while self.running:
            with QMutexLocker(self.mutex):
                is_paused = self.paused
            if is_paused:
                self.msleep(100)
                continue

            model_path_local, confidence_local, frame_local = None, None, None
            with QMutexLocker(self.mutex):
                if self.model_path_to_load:
                    model_path_local = self.model_path_to_load
                    self.model_path_to_load = None
                elif self.confidence_to_set is not None:
                    confidence_local = self.confidence_to_set
                    self.confidence_to_set = None
                elif self.current_frame is not None:
                    frame_local = self.current_frame
                    self.current_frame = None

            try:
                # --- START: Synchronous model loading logic ---
                if model_path_local:
                    print("[ZMQ Thread] Sending LOAD_MODEL command and waiting for confirmation...")
                    self.socket.send_string(f"LOAD_MODEL::{model_path_local}")
                    if self.socket.poll(5000, zmq.POLLIN): # 5-second timeout for model loading
                        response_bytes = self.socket.recv()
                        print(f"[ZMQ Thread] Received model load confirmation: {response_bytes.decode()}")
                    else:
                        print("[ZMQ Thread] Timeout waiting for model load confirmation.")
                    continue # Skip the rest of the loop to ensure no frames are sent
                # --- END: Synchronous model loading logic ---

                if confidence_local is not None:
                    self.socket.send_string(f"SET_CONFIDENCE::{confidence_local}")
                elif frame_local is not None:
                    _, buffer = cv2.imencode('.jpg', frame_local)
                    self.socket.send(buffer.tobytes())
                
                if self.socket.poll(20, zmq.POLLIN):
                    response_bytes = self.socket.recv()
                    response_data = json.loads(response_bytes.decode('utf-8'))
                    
                    if response_data.get('status') == 'success':
                        if 'detections_list' in response_data:
                            self.detections_ready.emit(response_data['detections_list'], response_data.get('inference_ms', 0))
                        
                        part_status = response_data.get('part_status', 'FAIL')
                        clip_status = response_data.get('clip_status', 'FAIL')
                        self.detection_statuses.emit(part_status, clip_status)
                        
                        total_status = "PASS" if part_status == 'PASS' and clip_status == 'PASS' else 'FAIL'
                        count = response_data.get('detections', 0)
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        history_msg = f"[{timestamp}] {total_status} - {count} objects detected."
                        self.detection_result.emit(history_msg)

                if frame_local is not None: self.frame_pending = False
            except (zmq.error.ZMQError, json.JSONDecodeError) as e:
                print(f"[ZMQ Thread] Error: {e}. Resetting connection.")
                self.create_and_connect_socket()
                self.frame_pending = False
            
            if model_path_local is None and confidence_local is None and frame_local is None:
                self.msleep(5)
                
        if self.socket: self.socket.close()
        print("[ZMQ Thread] Stopped.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    splash_pix = QPixmap("icons/pk_icon.png")
    if not splash_pix.isNull():
        splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
        splash.setMask(splash_pix.mask())
        splash.show()
        t_start = time.time()
        while time.time() < t_start + 1.5: app.processEvents()
    else: splash = None
    main_win = PokaiVisionUI()
    main_win.showFullScreen()
    if splash: splash.finish(main_win)
    sys.exit(app.exec_())

