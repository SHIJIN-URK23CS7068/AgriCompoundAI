import random
from utils.config import HW_MODE
from utils.logger import logger

class EnvironmentSensor:
    def __init__(self):
        logger.info(f"Initializing Environment Sensor (BME280) in [{HW_MODE.upper()}] mode")
        self.i2c_bus = None
        self.sensor_address = 0x76 # Common BME280 I2C address

        if HW_MODE == "pi":
            self._setup_hardware()

    def _setup_hardware(self):
        try:
            import smbus2
            import bme280
            self.smbus2 = smbus2
            self.bme280 = bme280
            self.i2c_bus = smbus2.SMBus(1) # Default Pi I2C bus
            # Read calibration data
            self.calibration_params = bme280.load_calibration_params(self.i2c_bus, self.sensor_address)
            logger.info("Successfully calibrated and connected BME280 sensor.")
        except Exception as e:
            logger.warning(f"Failed to connect physical BME280 sensor ({e}). Falling back to simulation.")
            self.i2c_bus = None

    def read_conditions(self) -> dict:
        """
        Reads temperature (Celsius) and relative humidity (%).
        Returns:
            dict: {"temperature": float, "humidity": float}
        """
        if HW_MODE == "pi" and self.i2c_bus is not None:
            return self._read_hardware()
        else:
            return self._read_simulated()

    def _read_hardware(self) -> dict:
        try:
            data = self.bme280.sample(self.i2c_bus, self.sensor_address, self.calibration_params)
            logger.info(f"BME280 physical reading: {data.temperature:.2f} C, {data.humidity:.2f} %RH")
            return {
                "temperature": round(data.temperature, 2),
                "humidity": round(data.humidity, 2)
            }
        except Exception as e:
            logger.error(f"Error reading physical BME280 sensor: {e}. Falling back to simulation.")
            return self._read_simulated()

    def _read_simulated(self) -> dict:
        # Simulate standard laboratory conditions
        # Adding a bit of random walk/fluctuation
        temp = round(23.5 + random.uniform(-1.5, 1.5), 2)
        humidity = round(55.0 + random.uniform(-5.0, 5.0), 2)
        logger.info(f"Simulated ambient conditions: {temp} C, {humidity} %RH")
        return {
            "temperature": temp,
            "humidity": humidity
        }
