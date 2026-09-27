import os
import sys
from pathlib import Path
from flask import Flask

# Add current folder to sys.path to ensure local imports load cleanly
sys.path.append(str(Path(__file__).resolve().parent))

from utils.config import BASE_DIR, HW_MODE, CLASSIFIER_PATH, REGRESSOR_PATH, LSTM_PATH
from utils.logger import logger

def create_app():
    logger.info("Initializing AgriCompoundAI Flask Application...")
    
    # Configure Flask app pointing to our web folder paths
    app = Flask(
        __name__,
        static_folder=str(BASE_DIR / "web" / "static"),
        template_folder=str(BASE_DIR / "web" / "templates")
    )
    
    # Enable registering blueprint
    from web.routes import web_bp
    app.register_blueprint(web_bp)
    
    # Run setup validation
    verify_system_setup()

    return app

def verify_system_setup():
    logger.info(f"Checking AgriCompoundAI pipeline configuration...")
    logger.info(f"System Hardware Mode: [{HW_MODE.upper()}]")
    
    # Verify weights
    weights_missing = False
    for path, name in [
        (CLASSIFIER_PATH, "Classifier"),
        (REGRESSOR_PATH, "Regressor"),
        (LSTM_PATH, "Shelf Life LSTM")
    ]:
        if not path.exists():
            logger.warning(f"Trained weight file for {name} missing at: {path}")
            weights_missing = True
        else:
            logger.info(f"Found weight file for {name} at: {path}")
            
    if weights_missing:
        logger.warning(
            "=== SYSTEM WARNING ===\n"
            "One or more model weights files are missing. The predictor will automatically "
            "run in demo fallback mode using uninitialized PyTorch backbones combined with "
            "color-space & index heuristics. The dashboard will work out-of-the-box.\n"
            "To train actual model weights, execute: python train.py"
        )
    else:
        logger.info("All model weights successfully verified. Ready for deep learning inference.")

if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on http://localhost:{port}...")
    
    # Run server
    app.run(host="0.0.0.0", port=port, debug=True)
