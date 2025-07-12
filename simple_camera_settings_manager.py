#!/usr/bin/env python3
"""
Simple camera settings manager that uses v4l2-ctl for camera controls.
This version avoids "device busy" errors by using v4l2-ctl instead of OpenCV for settings.
"""
import cv2
import json
import os
import subprocess

class SimpleCameraSettingsManager:
    """Simple camera settings manager using v4l2-ctl"""
    
    def __init__(self, device_path):
        self.device_path = device_path
        
        # Check if v4l2-ctl is available
        self.has_v4l2_ctl = self._check_v4l2_ctl()
        
        # OpenCV property mapping - fallback only
        self.prop_map = {
            "brightness": cv2.CAP_PROP_BRIGHTNESS,
            "contrast": cv2.CAP_PROP_CONTRAST,
            "saturation": cv2.CAP_PROP_SATURATION,
            "gain": cv2.CAP_PROP_GAIN,
            "sharpness": cv2.CAP_PROP_SHARPNESS,
            "white_balance_automatic": cv2.CAP_PROP_AUTO_WB,
        }
        
        # Default values for UI ranges
        self.default_values = {
            "brightness": 0,
            "contrast": 32,
            "saturation": 64,
            "gain": 0,
            "sharpness": 3,
            "white_balance_automatic": 1
        }
        
        print(f"Simple camera settings manager for {device_path}")
        print(f"v4l2-ctl available: {self.has_v4l2_ctl}")
    
    def _check_v4l2_ctl(self):
        """Check if v4l2-ctl is available"""
        try:
            result = subprocess.run(['v4l2-ctl', '--version'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def _run_v4l2_command(self, cmd):
        """Run a v4l2-ctl command"""
        try:
            full_cmd = f"v4l2-ctl -d {self.device_path} {cmd}"
            result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=5)
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)
    
    def _get_v4l2_control(self, control):
        """Get a v4l2 control value"""
        if not self.has_v4l2_ctl:
            return None
        
        success, stdout, stderr = self._run_v4l2_command(f"--get-ctrl {control}")
        if success:
            # Parse output like "brightness: 64"
            for line in stdout.strip().split('\n'):
                if ':' in line:
                    try:
                        return int(line.split(':')[1].strip())
                    except:
                        return None
        return None
    
    def _set_v4l2_control(self, control, value):
        """Set a v4l2 control using v4l2-ctl"""
        if not self.has_v4l2_ctl:
            return False
        
        success, stdout, stderr = self._run_v4l2_command(f"--set-ctrl {control}={value}")
        if success:
            print(f"✓ Set {control}={value} for {self.device_path}")
            return True
        else:
            print(f"✗ Failed to set {control}={value} for {self.device_path}: {stderr}")
            return False
    
    def _set_multiple_v4l2_controls(self, controls_dict):
        """Set multiple v4l2 controls in a single command for better performance"""
        if not self.has_v4l2_ctl or not controls_dict:
            return {}
        
        # Build a single command with multiple controls
        control_args = []
        for control, value in controls_dict.items():
            control_args.append(f"{control}={value}")
        
        if not control_args:
            return {}
        
        # Execute single command with all controls
        controls_str = ",".join(control_args)
        success, stdout, stderr = self._run_v4l2_command(f"--set-ctrl {controls_str}")
        
        results = {}
        if success:
            print(f"✓ Batch set controls for {self.device_path}: {controls_str}")
            for control in controls_dict.keys():
                results[control] = True
        else:
            print(f"✗ Batch set failed for {self.device_path}: {stderr}")
            # Fall back to individual calls if batch fails
            for control, value in controls_dict.items():
                results[control] = self._set_v4l2_control(control, value)
        
        return results
    
    def _get_multiple_v4l2_controls(self, controls_list):
        """Get multiple v4l2 controls in a single command for better performance"""
        if not self.has_v4l2_ctl or not controls_list:
            return {}
        
        # Build a single command with multiple controls
        controls_str = ",".join(controls_list)
        success, stdout, stderr = self._run_v4l2_command(f"--get-ctrl {controls_str}")
        
        results = {}
        if success:
            # Parse output like "brightness: 64\ncontrast: 32\n"
            for line in stdout.strip().split('\n'):
                if ':' in line:
                    try:
                        control, value = line.split(':', 1)
                        results[control.strip()] = int(value.strip())
                    except:
                        pass
        
        # Fill in missing controls with individual calls
        for control in controls_list:
            if control not in results:
                value = self._get_v4l2_control(control)
                if value is not None:
                    results[control] = value
        
        return results

    def _get_camera(self):
        """Get camera instance - try different methods like the app does (fallback only)"""
        try:
            # Extract device index from path like /dev/video0 -> 0
            device_index = int(''.join(filter(str.isdigit, self.device_path)))
            
            # Try by index first
            cap = cv2.VideoCapture(device_index)
            if cap.isOpened():
                return cap
            
            # Try by path
            cap = cv2.VideoCapture(self.device_path)
            if cap.isOpened():
                return cap
                
            # Try with V4L2 backend
            cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
            if cap.isOpened():
                return cap
                
        except Exception as e:
            print(f"Error opening camera {self.device_path}: {e}")
        
        return None
    
    def get_available_controls(self):
        """Get list of available controls - return all basic ones"""
        # Since your cameras are working, provide the basic controls
        return ["brightness", "contrast", "saturation", "gain", "sharpness", "white_balance_automatic"]
    
    def get_control(self, control_name):
        """Get a camera control value using v4l2-ctl first, then OpenCV fallback"""
        # Try v4l2-ctl first (preferred method)
        if self.has_v4l2_ctl:
            value = self._get_v4l2_control(control_name)
            if value is not None:
                return value
        
        # Fallback to OpenCV (may cause device busy errors)
        if control_name not in self.prop_map:
            return self.default_values.get(control_name, 0)
        
        cap = self._get_camera()
        if cap:
            try:
                prop = self.prop_map[control_name]
                value = cap.get(prop)
                cap.release()
                
                # Convert OpenCV values to UI ranges like before
                if value is not None and value != -1:
                    if control_name == "brightness":
                        # Convert OpenCV brightness to UI range (-64 to 64)
                        return int((value - 0.5) * 128)
                    elif control_name == "contrast":
                        # Convert OpenCV contrast to UI range (0 to 64)
                        return int(value * 64)
                    elif control_name == "saturation":
                        # Convert OpenCV saturation to UI range (0 to 128)
                        return int(value * 128)
                    else:
                        return int(value)
            except Exception as e:
                print(f"Error getting {control_name} via OpenCV: {e}")
                if cap:
                    cap.release()
        
        # Return default value if we can't get the actual value
        return self.default_values.get(control_name, 0)
    
    def set_control(self, control_name, value):
        """Set a camera control value using v4l2-ctl first, then OpenCV fallback"""
        # Try v4l2-ctl first (preferred method)
        if self.has_v4l2_ctl:
            success = self._set_v4l2_control(control_name, value)
            if success:
                return True
        
        # Fallback to OpenCV (may cause device busy errors)
        if control_name not in self.prop_map:
            return False
        
        cap = self._get_camera()
        if cap:
            try:
                prop = self.prop_map[control_name]
                
                # Convert UI values to OpenCV ranges
                cv_value = value
                if control_name == "brightness":
                    cv_value = (value / 128.0) + 0.5  # Convert from -64..64 to 0..1
                elif control_name == "contrast":
                    cv_value = value / 64.0  # Convert from 0..64 to 0..1
                elif control_name == "saturation":
                    cv_value = value / 128.0  # Convert from 0..128 to 0..1
                
                success = cap.set(prop, cv_value)
                cap.release()
                
                if success:
                    print(f"✓ Set {control_name}={value} for {self.device_path} via OpenCV")
                    return True
                else:
                    print(f"✗ Failed to set {control_name}={value} for {self.device_path} via OpenCV")
                    
            except Exception as e:
                print(f"✗ Error setting {control_name}={value} via OpenCV: {e}")
                if cap:
                    cap.release()
        
        return False

    def get_all_controls(self):
        """Get all available controls at once for better performance"""
        controls = self.get_available_controls()
        
        # Try batch operation first
        if self.has_v4l2_ctl:
            batch_results = self._get_multiple_v4l2_controls(controls)
            if batch_results:
                return batch_results
        
        # Fall back to individual calls
        results = {}
        for control in controls:
            value = self.get_control(control)
            if value is not None:
                results[control] = value
        
        return results
    
    def set_all_controls(self, controls_dict):
        """Set all controls at once for better performance"""
        if not controls_dict:
            return {}
        
        # Try batch operation first
        if self.has_v4l2_ctl:
            batch_results = self._set_multiple_v4l2_controls(controls_dict)
            if batch_results and all(batch_results.values()):
                return batch_results
        
        # Fall back to individual calls
        results = {}
        for control, value in controls_dict.items():
            results[control] = self.set_control(control, value)
        
        return results

# Create alias for backward compatibility
CameraSettingsManager = SimpleCameraSettingsManager
EnhancedCameraSettingsManager = SimpleCameraSettingsManager

def test_simple_camera_settings():
    """Test the simple camera settings manager"""
    print("=== Testing Simple Camera Settings Manager ===")
    
    manager = SimpleCameraSettingsManager("/dev/video0")
    
    # Get available controls
    controls = manager.get_available_controls()
    print(f"Available controls: {controls}")
    
    # Test getting current values
    print("Current values:")
    for control in controls:
        value = manager.get_control(control)
        print(f"  {control}: {value}")
    
    # Test setting a value
    if "brightness" in controls:
        old_value = manager.get_control("brightness")
        test_value = 10
        success = manager.set_control("brightness", test_value)
        new_value = manager.get_control("brightness")
        print(f"Test: brightness {old_value} -> {test_value} (actual: {new_value}) [{'SUCCESS' if success else 'FAILED'}]")

if __name__ == "__main__":
    test_simple_camera_settings()

