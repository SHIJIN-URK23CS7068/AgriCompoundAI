import numpy as np
from utils.config import WAVELENGTHS
from utils.logger import logger

class SpectralPreprocessor:
    def __init__(self):
        self.wavelengths = WAVELENGTHS

    def preprocess(self, raw_spectrum: dict) -> dict:
        """
        Processes raw spectral reflectance values, calculates key vegetation/quality indices.
        Returns:
            dict: {
                "vector": np.ndarray (18,),
                "indices": {
                    "NDVI": float,  # Normalized Difference Vegetation Index
                    "WBI": float,   # Water Band Index (moisture status)
                    "CI": float,    # Chlorophyll Index
                    "CurcuminIndex": float,
                    "AnthocyaninIndex": float
                }
            }
        """
        # Ensure all wavelengths are present
        spectrum = {wl: raw_spectrum.get(wl, 0.0) for wl in self.wavelengths}
        
        # Build vector
        vector = np.array([spectrum[wl] for wl in self.wavelengths], dtype=np.float32)

        # Helper to get wavelength value safely
        def val(wl):
            return spectrum.get(wl, 0.01)

        # 1. NDVI (Normalized Difference Vegetation Index)
        # Formula: (NIR - Red) / (NIR + Red)
        # Red = 680nm, NIR = 760nm or 810nm
        nir_val = val(760)
        red_val = val(680)
        ndvi = (nir_val - red_val) / max(0.001, (nir_val + red_val))

        # 2. WBI (Water Band Index)
        # Formula: 900nm / 940nm (sensitive to water content)
        wbi = val(900) / max(0.001, val(940))

        # 3. CI (Chlorophyll Index / Greenness)
        # Formula: (NIR / Red) - 1
        ci = (val(760) / max(0.001, val(680))) - 1

        # 4. Curcumin Index (custom ratio for yellow pigment Curcumin)
        # Curcumin absorbs blue (460nm) and reflects yellow/orange (585nm)
        curcumin_idx = (val(585) - val(460)) / max(0.001, (val(585) + val(460)))

        # 5. Anthocyanin Index (custom ratio for purple/red pigment Anthocyanin)
        # Anthocyanin absorbs green (535nm) and reflects blue/red/NIR (760nm)
        anthocyanin_idx = (val(760) / max(0.001, val(535)))

        indices = {
            "NDVI": float(np.clip(ndvi, -1.0, 1.0)),
            "WBI": float(np.clip(wbi, 0.0, 5.0)),
            "CI": float(np.clip(ci, -1.0, 10.0)),
            "CurcuminIndex": float(np.clip(curcumin_idx, -1.0, 1.0)),
            "AnthocyaninIndex": float(np.clip(anthocyanin_idx, 0.0, 20.0))
        }

        logger.debug(f"Computed spectral indices: {indices}")
        return {
            "vector": vector,
            "indices": indices
        }
