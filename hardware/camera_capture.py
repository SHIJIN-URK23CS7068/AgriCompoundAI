import cv2
import numpy as np
import os
from pathlib import Path
from utils.config import HW_MODE, IMAGE_SIZE, BASE_DIR
from utils.logger import logger

class CameraCapture:
    def __init__(self):
        logger.info(f"Initializing Camera Capture in [{HW_MODE.upper()}] mode")
        self.static_img_dir = BASE_DIR / "web" / "static" / "images"

    def capture_image(self, plant_name: str = None) -> np.ndarray:
        """
        Captures an RGB/BGR image.
        On Pi: Captures from Pi Camera Module 3 or webcam.
        On Sim: Loads the high-quality simulated image for the given plant name.
        """
        if HW_MODE == "pi":
            return self._capture_hardware()
        else:
            return self._capture_simulated(plant_name)

    def _capture_hardware(self) -> np.ndarray:
        logger.info("Attempting to capture image from hardware camera...")
        
        # Try picamera2 (Pi Camera Module 3 standard library)
        try:
            from picamera2 import Picamera2
            picam = Picamera2()
            picam.configure(picam.create_preview_configuration())
            picam.start()
            # Capture frame
            frame = picam.capture_array()
            picam.stop()
            picam.close()
            # Picamera2 outputs RGB by default, convert to BGR for OpenCV consistency
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            logger.info("Successfully captured image using picamera2.")
            return bgr_frame
        except Exception as e:
            logger.warning(f"picamera2 capture failed ({e}), falling back to OpenCV VideoCapture.")
            
        # Fallback to OpenCV camera capture (USB webcam or standard interface)
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Could not open video device 0")
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                logger.info("Successfully captured image using OpenCV VideoCapture.")
                return frame
            else:
                raise RuntimeError("OpenCV read returned empty frame")
        except Exception as e:
            logger.error(f"Hardware camera capture completely failed: {e}. Falling back to default pattern.")
            return self._create_fallback_pattern("Unknown")

    def _capture_simulated(self, plant_name: str) -> np.ndarray:
        if not plant_name:
            plant_names = ["Tomato", "Spinach", "Turmeric", "Basil", "Lavender"]
            plant_name = np.random.choice(plant_names)
            logger.info(f"No plant name specified for simulation capture. Randomly selected: {plant_name}")
        
        logger.info(f"Simulating scan for plant: {plant_name}")
        
        img_path = self.static_img_dir / f"{plant_name.lower()}.png"
        
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                logger.info(f"Loaded simulated image from {img_path}")
                return img
            else:
                logger.warning(f"Could not read image file {img_path}")
        else:
            logger.warning(f"Simulated image file {img_path} does not exist.")
            
        return self._create_fallback_pattern(plant_name)

    def _create_fallback_pattern(self, plant_name: str) -> np.ndarray:
        logger.info(f"Creating fallback synthetic pattern for {plant_name}")
        # Draw a synthetic colored circle/pattern
        img = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
        img.fill(40) # Dark gray background
        
        color_map = {
            "Tomato": (0, 0, 220),       # Red
            "Spinach": (0, 180, 0),     # Green
            "Turmeric": (0, 120, 220),   # Orange-ish BGR
            "Basil": (0, 220, 0),       # Bright Green
            "Lavender": (180, 50, 180), # Purple
        }
        
        color = color_map.get(plant_name, (200, 200, 200))
        
        # Draw a shape representing the plant
        cv2.circle(img, (112, 112), 70, color, -1)
        # Add some text
        cv2.putText(img, f"SIM: {plant_name}", (15, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add noise
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img
