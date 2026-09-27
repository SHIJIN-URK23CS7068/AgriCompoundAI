import os
import cv2
import uuid
from datetime import datetime
from flask import Blueprint, render_template, jsonify, request, current_app
from pathlib import Path

from utils.config import BASE_DIR, HW_MODE
from utils.logger import logger
from hardware.camera_capture import CameraCapture
from hardware.environment_sensor import EnvironmentSensor
from multispectral.sensor_reader import MultispectralSensorReader
from inference.predictor import AgriPredictor
from xai.explainability import AgriExplainability
from database.db_manager import DBManager

web_bp = Blueprint("web", __name__, template_folder="templates")

# Initialize pipeline modules
camera = CameraCapture()
env_sensor = EnvironmentSensor()
spectral_sensor = MultispectralSensorReader()
predictor = AgriPredictor()
explainer = AgriExplainability()
db = DBManager()

# Ensure temp folders for scans exist under static/captures
CAPTURES_DIR = BASE_DIR / "web" / "static" / "captures"
os.makedirs(CAPTURES_DIR, exist_ok=True)

@web_bp.route("/")
def index():
    return render_template("index.html", hw_mode=HW_MODE)

@web_bp.route("/scan", methods=["POST"])
def scan():
    try:
        # 1. Parse requested plant in simulation mode
        data = request.get_json() or {}
        requested_plant = data.get("plant_name", None)
        
        logger.info(f"Scan triggered. Target plant: {requested_plant}")
        
        # 2. Capture Image
        img_bgr = camera.capture_image(plant_name=requested_plant)
        
        # 3. Read environment temperature and humidity
        env = env_sensor.read_conditions()
        
        # 4. Get historical scans for this plant to feed into the LSTM (last 4 scans)
        # We need the plant name to fetch history. If requested_plant is None, we'll
        # run a first-pass rule-based classification to find the history, or fetch generic.
        # For simplicity, we query database history based on the requested_plant or overall.
        history = db.get_history(plant_name=requested_plant, limit=5)
        
        # 5. Read Spectrum (using temporary freshness guess for simulation accuracy)
        temp_freshness = 90.0
        if history and len(history) > 0:
            temp_freshness = history[0].get("freshness", 90.0)
        
        raw_spectrum = spectral_sensor.read_spectrum(plant_name=requested_plant or "Tomato", freshness=temp_freshness)
        
        # 6. Run unified predictions (PyTorch Classifier, Regressor, LSTM)
        # Pass raw spectrum, color features, and history list for sequence LSTM
        results = predictor.predict(
            img_bgr=img_bgr,
            raw_spectrum=raw_spectrum,
            env_conditions=env,
            history_list=history
        )
        
        # Identified plant name
        detected_plant = results["plant_name"]
        
        # 7. Generate Grad-CAM heatmap
        heatmap_img = explainer.generate_heatmap(img_bgr, detected_plant)
        
        # Generate plain-language XAI report
        report = explainer.generate_explanation(results, env)
        results["explanation_report"] = report

        # 8. Save images to disk
        scan_id = str(uuid.uuid4())[:8]
        raw_filename = f"raw_{scan_id}.png"
        heatmap_filename = f"heatmap_{scan_id}.png"
        
        raw_path = CAPTURES_DIR / raw_filename
        heatmap_path = CAPTURES_DIR / heatmap_filename
        
        cv2.imwrite(str(raw_path), img_bgr)
        cv2.imwrite(str(heatmap_path), heatmap_img)
        
        # Web-accessible paths
        web_raw_path = f"/static/captures/{raw_filename}"
        web_heatmap_path = f"/static/captures/{heatmap_filename}"
        
        # 9. Add database entries
        scan_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "plant_name": detected_plant,
            "scientific_name": results["scientific_name"],
            "confidence": results["confidence"],
            "freshness": results["freshness"],
            "quality": results["quality"],
            "grade": results["grade"],
            "maturity": results["maturity"],
            "maturity_label": results["maturity_label"],
            "shelf_life": results["shelf_life"],
            "temperature": env["temperature"],
            "humidity": env["humidity"],
            "ndvi": results["indices"]["NDVI"],
            "wbi": results["indices"]["WBI"],
            "raw_spectrum": raw_spectrum,
            "compounds": results["compounds"],
            "image_path": web_raw_path,
            "heatmap_path": web_heatmap_path
        }
        
        db_id = db.insert_scan(scan_record)
        results["db_id"] = db_id
        results["image_path"] = web_raw_path
        results["heatmap_path"] = web_heatmap_path
        results["timestamp"] = scan_record["timestamp"]
        results["temperature"] = env["temperature"]
        results["humidity"] = env["humidity"]
        results["raw_spectrum"] = raw_spectrum
        
        return jsonify({"status": "success", "data": results})
        
    except Exception as e:
        logger.error(f"Error executing scan endpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

@web_bp.route("/history", methods=["GET"])
def history():
    plant_name = request.args.get("plant_name", None)
    scans = db.get_history(plant_name=plant_name, limit=50)
    return jsonify({"status": "success", "data": scans})

@web_bp.route("/clear_history", methods=["POST"])
def clear_history():
    db.clear_history()
    # Clear captures directory files
    try:
        for file in os.listdir(CAPTURES_DIR):
            file_path = CAPTURES_DIR / file
            if os.path.isfile(file_path):
                os.unlink(file_path)
        logger.info("Cleared scan captures folder.")
    except Exception as e:
        logger.warning(f"Failed to clear captures folder files: {e}")
    return jsonify({"status": "success", "message": "Scans database and file history cleared."})
