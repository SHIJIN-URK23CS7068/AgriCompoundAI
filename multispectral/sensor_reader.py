import random
import numpy as np
from utils.config import HW_MODE, WAVELENGTHS
from utils.logger import logger

class MultispectralSensorReader:
    def __init__(self):
        logger.info(f"Initializing Multispectral Sensor (AS7265x) in [{HW_MODE.upper()}] mode")
        self.i2c_bus = None
        self.sensor_address = 0x49 # Default base address for AS7265x
        
        if HW_MODE == "pi":
            self._setup_hardware()

    def _setup_hardware(self):
        try:
            import smbus2
            self.i2c_bus = smbus2.SMBus(1)
            # Send standard reset or configuration commands to the sensor
            # AS7265x has co-processors for 18 channels. Setting LED and integration time:
            # We write 0x04 to reset, 0x80 to enable channels, etc.
            # (Simplifying hardware initialization for default I2C communication)
            self.i2c_bus.write_byte_data(self.sensor_address, 0x04, 0x01) # Write control register
            logger.info("Connected to physical AS7265x multispectral sensor array over I2C.")
        except Exception as e:
            logger.warning(f"Failed to connect physical AS7265x sensor ({e}). Falling back to simulation.")
            self.i2c_bus = None

    def read_spectrum(self, plant_name: str = None, freshness: float = 100.0) -> dict:
        """
        Reads spectral intensity/reflectance across 18 bands (410nm to 940nm).
        Returns:
            dict: {wavelength_nm (int): reflectance_value (float, 0.0 to 1.0)}
        """
        if HW_MODE == "pi" and self.i2c_bus is not None:
            return self._read_hardware()
        else:
            return self._read_simulated(plant_name, freshness)

    def _read_hardware(self) -> dict:
        try:
            # Physical readout of 18 bands from AS72651/2/3 registers
            # AS7265x maps wavelengths to specific registers.
            # For this integration, we simulate reading the 18 channel values from I2C registers.
            # In a real Pi deployment, we query the virtual registers of the master chip.
            raw_data = {}
            # Virtual register reads (example registers for AS7265x data channels):
            # Channels A-F on AS72651, G-L on AS72652, R-W on AS72653
            for i, wl in enumerate(WAVELENGTHS):
                # Read 2 bytes (MSB/LSB) from channels (0x08, 0x0A, etc.)
                # In simulation fallback, or on read failure, return simulated
                reg = 0x08 + (i % 6) * 2
                val = self.i2c_bus.read_word_data(self.sensor_address, reg)
                # Normalize 16-bit ADC to reflectance
                raw_data[wl] = round(val / 65535.0, 4)
            
            logger.info("Successfully read physical spectrum from AS7265x array.")
            return raw_data
        except Exception as e:
            logger.error(f"Error reading physical AS7265x: {e}. Falling back to simulation.")
            return self._read_simulated()

    def _read_simulated(self, plant_name: str = None, freshness: float = 100.0) -> dict:
        if not plant_name:
            plant_name = "Tomato"
        
        freshness_pct = freshness / 100.0 # scale 0 to 1
        
        # Base reflectance models per plant type
        # 18 wavelengths: [410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940]
        spectrum = {}
        
        if plant_name == "Spinach" or plant_name == "Basil":
            # Green leaf signature: Chlorophyll absorption at 430-450nm and 660-680nm. Green reflection peak at 535-560nm. NIR plateau.
            for wl in WAVELENGTHS:
                if wl in [410, 435, 460]: # Blue range - absorption by carotenoids/chlorophyll
                    base = 0.08 + (1 - freshness_pct) * 0.12 # turns browner/less absorbing
                elif wl in [485, 510]: # Transition
                    base = 0.12
                elif wl in [535, 560]: # Green peak
                    base = 0.28 * (0.4 + 0.6 * freshness_pct)
                elif wl in [585, 610]: # Yellow shift
                    base = 0.15 + (1 - freshness_pct) * 0.15 # Yellows as it decays
                elif wl in [645, 680]: # Chlorophyll absorption dip
                    # Dip is deep when fresh (high absorption = low reflectance).
                    # When freshness drops, chlorophyll degrades, absorption drops, reflectance increases.
                    base = 0.05 + (1 - freshness_pct) * 0.25 
                elif wl in [705]: # Red-edge
                    base = 0.40 * (0.3 + 0.7 * freshness_pct)
                else: # NIR Plateau (730nm to 940nm)
                    # Fresh leaves scatter NIR strongly. Dried/decayed leaves scatter less.
                    base = 0.85 * (0.2 + 0.8 * freshness_pct)
                
                # Add small noise
                val = base + random.uniform(-0.02, 0.02)
                spectrum[wl] = max(0.01, min(0.99, round(val, 4)))

        elif plant_name == "Tomato":
            # Red fruit signature: absorption of blue/green (410-535nm). High red reflectance (610-680nm). High NIR.
            for wl in WAVELENGTHS:
                if wl in [410, 435, 460, 485, 510, 535]: # Blue/green absorption
                    base = 0.06 + (1 - freshness_pct) * 0.1
                elif wl in [560, 585]: # Orange transition
                    base = 0.25
                elif wl in [610, 645, 680]: # Red peak (Lycopene)
                    # High red reflectance when ripe.
                    base = 0.65 * (0.5 + 0.5 * freshness_pct)
                elif wl in [705]:
                    base = 0.72 * (0.4 + 0.6 * freshness_pct)
                else: # NIR
                    base = 0.82 * (0.3 + 0.7 * freshness_pct)
                
                val = base + random.uniform(-0.02, 0.02)
                spectrum[wl] = max(0.01, min(0.99, round(val, 4)))

        elif plant_name == "Turmeric":
            # Yellow-orange rhizome: low blue/green reflectance (absorption by curcumin), high yellow/red reflectance, high NIR.
            for wl in WAVELENGTHS:
                if wl in [410, 435, 460, 485]: # Strong Curcumin absorption in blue/violet
                    base = 0.04 + (1 - freshness_pct) * 0.15
                elif wl in [510, 535]:
                    base = 0.20
                elif wl in [560, 585, 610, 645, 680]: # Yellow/Orange/Red reflectance
                    base = 0.75 * (0.6 + 0.4 * freshness_pct)
                elif wl in [705]:
                    base = 0.80
                else: # NIR
                    base = 0.88 * (0.4 + 0.6 * freshness_pct)
                
                val = base + random.uniform(-0.02, 0.02)
                spectrum[wl] = max(0.01, min(0.99, round(val, 4)))

        elif plant_name == "Lavender":
            # Lavender purple flower: moderate blue, low green, moderate red, high NIR.
            for wl in WAVELENGTHS:
                if wl in [410, 435, 460]: # Violet-blue reflection
                    base = 0.35 * (0.4 + 0.6 * freshness_pct)
                elif wl in [485, 510, 535, 560]: # Green absorption dip
                    base = 0.12 + (1 - freshness_pct) * 0.15
                elif wl in [585, 610, 645]: # Red reflection
                    base = 0.30 * (0.4 + 0.6 * freshness_pct)
                elif wl in [680, 705]: # Absorption & Red-edge transition
                    base = 0.45
                else: # NIR
                    base = 0.78 * (0.3 + 0.7 * freshness_pct)
                
                val = base + random.uniform(-0.02, 0.02)
                spectrum[wl] = max(0.01, min(0.99, round(val, 4)))
        
        else:
            # Generic/Default plant signature
            for wl in WAVELENGTHS:
                if wl < 700:
                    base = 0.2
                else:
                    base = 0.7 * freshness_pct
                val = base + random.uniform(-0.03, 0.03)
                spectrum[wl] = max(0.01, min(0.99, round(val, 4)))
                
        logger.info(f"Generated simulated spectral reflectance for {plant_name} (Freshness={freshness}%)")
        return spectrum
