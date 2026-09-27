import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
DB_PATH = DATA_DIR / "agri_compound.db"
LOG_PATH = DATA_DIR / "system.log"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Hardware Configuration
# "sim" for simulation fallback, "pi" for physical Raspberry Pi
HW_MODE = os.environ.get("AGRICOMPOUND_HW_MODE", "sim").lower()
if HW_MODE not in ["sim", "pi"]:
    HW_MODE = "sim"

# Image and Camera Config
IMAGE_SIZE = (224, 224)
TOUCHSCREEN_RESOLUTION = (800, 480)

# Spectral Configuration (AS7265x 18 Wavelengths in nm)
WAVELENGTHS = [410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940]

# Model Paths
CLASSIFIER_PATH = WEIGHTS_DIR / "classifier.pth"
REGRESSOR_PATH = WEIGHTS_DIR / "regressor.pth"
LSTM_PATH = WEIGHTS_DIR / "lstm.pth"

# Seed Knowledge Base for 5 Vegetables/Medicinal Plants
PLANT_KNOWLEDGE = {
    "Tomato": {
        "scientific_name": "Solanum lycopersicum",
        "nutrition": "Rich in Lycopene (antioxidant), Vitamin C, Potassium, and Folate.",
        "similar_plants": "Bell Pepper, Eggplant, Tomatillo",
        "storage_tips": "Store at room temperature away from direct sunlight. Do not refrigerate unripened tomatoes, as it degrades flavor and texture.",
        "educational_supplement": "Lycopene bioavailability increases when tomatoes are cooked with a small amount of healthy fats (e.g. olive oil).",
        "compounds": ["Lycopene", "Chlorophyll", "Beta-carotene"]
    },
    "Spinach": {
        "scientific_name": "Spinacia oleracea",
        "nutrition": "High in Iron, Calcium, Vitamins A, C, K1, and Folic Acid.",
        "similar_plants": "Kale, Swiss Chard, Mustard Greens",
        "storage_tips": "Store in a sealed container lined with a dry paper towel in the refrigerator crisper drawer. Keep dry to prevent wilting.",
        "educational_supplement": "Contains oxalic acid, which can inhibit calcium absorption. Light steaming reduces oxalic acid and improves mineral uptake.",
        "compounds": ["Chlorophyll", "Lutein", "Beta-carotene"]
    },
    "Turmeric": {
        "scientific_name": "Curcuma longa",
        "nutrition": "Contains Curcumin (powerful anti-inflammatory and antioxidant), Vitamin B6, and Manganese.",
        "similar_plants": "Ginger, Galangal, Cardamom",
        "storage_tips": "Store fresh rhizomes in a cool, dark, dry place in a paper bag, or refrigerate in an airtight container for up to 3 weeks.",
        "educational_supplement": "Curcumin absorption in the human body is enhanced by up to 2000% when consumed with black pepper (containing piperine).",
        "compounds": ["Curcumin", "Chlorophyll"]
    },
    "Basil": {
        "scientific_name": "Ocimum basilicum",
        "nutrition": "Contains Vitamin K, Eugenol (antioxidant), Linalool, and Iron.",
        "similar_plants": "Mint, Oregano, Sage, Thai Basil",
        "storage_tips": "Treat fresh basil like cut flowers: trim stems and place in a jar of water at room temperature. Do not refrigerate, as cold turns leaves black.",
        "educational_supplement": "Basil essential oils have antibacterial properties and have been traditionally used to support digestive health.",
        "compounds": ["Chlorophyll", "Rosmarinic acid", "Linalool"]
    },
    "Lavender": {
        "scientific_name": "Lavandula angustifolia",
        "nutrition": "High in Linalyl acetate, Linalool, and anthocyanin phytochemicals.",
        "similar_plants": "Rosemary, Thyme, Mint",
        "storage_tips": "Hang fresh bunches upside down in a warm, dark, well-ventilated space to dry. Store dried buds in airtight glass containers.",
        "educational_supplement": "Primarily used for its calming aroma, lavender tea is also consumed for relaxation and relieving digestive bloating.",
        "compounds": ["Anthocyanins", "Linalool", "Chlorophyll"]
    }
}

# Bioactive Compound Linear Calibration Formula Placeholder Coefficients
# Formula: Concentration = alpha + sum(beta_i * R_i) + gamma * ColorMetric
# R_i is reflectance at specific bands, ColorMetric is average HSV/Lab value.
# WARNING: These are unvalidated research approximations, NOT certified lab measurements.
BIOACTIVE_COEFFICIENTS = {
    "Tomato": {
        "Lycopene": {
            "alpha": 5.0,
            "spectral_weights": {645: 25.0, 680: -15.0}, # Red reflectance vs absorption
            "color_channel": "R",
            "color_weight": 0.1,
            "unit": "mg/100g"
        },
        "Chlorophyll": {
            "alpha": 12.0,
            "spectral_weights": {680: -30.0, 760: 20.0}, # Chlorophyll dip vs NIR rise
            "color_channel": "G",
            "color_weight": 0.05,
            "unit": "mg/100g"
        }
    },
    "Spinach": {
        "Chlorophyll": {
            "alpha": 45.0,
            "spectral_weights": {680: -80.0, 760: 60.0},
            "color_channel": "G",
            "color_weight": 0.3,
            "unit": "mg/100g"
        },
        "Beta-carotene": {
            "alpha": 6.0,
            "spectral_weights": {485: -15.0, 535: 10.0},
            "color_channel": "R",
            "color_weight": 0.05,
            "unit": "mg/100g"
        }
    },
    "Turmeric": {
        "Curcumin": {
            "alpha": 120.0,
            "spectral_weights": {460: -150.0, 585: 100.0}, # Yellow vs Blue reflectance
            "color_channel": "R",
            "color_weight": 0.5,
            "unit": "mg/100g"
        }
    },
    "Basil": {
        "Chlorophyll": {
            "alpha": 35.0,
            "spectral_weights": {680: -60.0, 760: 45.0},
            "color_channel": "G",
            "color_weight": 0.2,
            "unit": "mg/100g"
        },
        "Rosmarinic acid": {
            "alpha": 15.0,
            "spectral_weights": {410: -20.0, 510: 15.0},
            "color_channel": "B",
            "color_weight": 0.05,
            "unit": "mg/100g"
        }
    },
    "Lavender": {
        "Anthocyanins": {
            "alpha": 8.0,
            "spectral_weights": {535: -40.0, 435: 25.0, 680: 15.0}, # Anthocyanin green-absorption
            "color_channel": "B",
            "color_weight": 0.1,
            "unit": "mg/100g"
        },
        "Linalool": {
            "alpha": 1.2,
            "spectral_weights": {900: 2.0, 940: -1.0},
            "color_channel": "R",
            "color_weight": 0.005,
            "unit": "g/100g"
        }
    }
}
