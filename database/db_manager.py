import sqlite3
import json
from datetime import datetime
from pathlib import Path
from utils.config import DB_PATH
from utils.logger import logger

class DBManager:
    def __init__(self):
        self.db_path = DB_PATH
        logger.info(f"Initializing SQLite Database at: {self.db_path}")
        self._create_tables()

    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                plant_name TEXT NOT NULL,
                scientific_name TEXT,
                confidence REAL,
                freshness REAL,
                quality REAL,
                grade TEXT,
                maturity REAL,
                maturity_label TEXT,
                shelf_life REAL,
                temperature REAL,
                humidity REAL,
                ndvi REAL,
                wbi REAL,
                raw_spectrum TEXT,
                compounds TEXT,
                image_path TEXT,
                heatmap_path TEXT
            )
        """)
        conn.commit()
        conn.close()
        logger.info("Database tables verified/created successfully.")

    def insert_scan(self, scan_data: dict) -> int:
        """
        Inserts a new scan result into the database.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        timestamp = scan_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Serialize dicts to JSON strings for SQLite storage
        raw_spectrum_json = json.dumps(scan_data.get("raw_spectrum", {}))
        compounds_json = json.dumps(scan_data.get("compounds", {}))
        
        try:
            cursor.execute("""
                INSERT INTO scans (
                    timestamp, plant_name, scientific_name, confidence, freshness, 
                    quality, grade, maturity, maturity_label, shelf_life, 
                    temperature, humidity, ndvi, wbi, raw_spectrum, 
                    compounds, image_path, heatmap_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                scan_data.get("plant_name"),
                scan_data.get("scientific_name"),
                scan_data.get("confidence"),
                scan_data.get("freshness"),
                scan_data.get("quality"),
                scan_data.get("grade"),
                scan_data.get("maturity"),
                scan_data.get("maturity_label"),
                scan_data.get("shelf_life"),
                scan_data.get("temperature"),
                scan_data.get("humidity"),
                scan_data.get("ndvi"),
                scan_data.get("wbi"),
                raw_spectrum_json,
                compounds_json,
                scan_data.get("image_path"),
                scan_data.get("heatmap_path")
            ))
            conn.commit()
            last_id = cursor.lastrowid
            logger.info(f"Logged scan result to database (ID: {last_id}) for {scan_data.get('plant_name')}")
            return last_id
        except Exception as e:
            logger.error(f"Failed to insert scan record: {e}")
            raise e
        finally:
            conn.close()

    def get_history(self, plant_name: str = None, limit: int = 50) -> list:
        """
        Queries scans history. Can filter by plant name.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM scans"
        params = []
        
        if plant_name:
            query += " WHERE plant_name = ?"
            params.append(plant_name)
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            history = []
            for row in rows:
                item = dict(row)
                # Deserialize JSON strings back to dicts
                item["raw_spectrum"] = json.loads(item["raw_spectrum"]) if item["raw_spectrum"] else {}
                item["compounds"] = json.loads(item["compounds"]) if item["compounds"] else {}
                history.append(item)
                
            return history
        except Exception as e:
            logger.error(f"Failed to query scan history: {e}")
            return []
        finally:
            conn.close()

    def get_scan_by_id(self, scan_id: int) -> dict:
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM scans WHERE id = ?", (scan_id,))
            row = cursor.fetchone()
            if row:
                item = dict(row)
                item["raw_spectrum"] = json.loads(item["raw_spectrum"]) if item["raw_spectrum"] else {}
                item["compounds"] = json.loads(item["compounds"]) if item["compounds"] else {}
                return item
            return None
        except Exception as e:
            logger.error(f"Failed to get scan by id {scan_id}: {e}")
            return None
        finally:
            conn.close()
            
    def clear_history(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM scans")
            conn.commit()
            logger.info("Cleared scans history from database.")
        except Exception as e:
            logger.error(f"Failed to clear scans history: {e}")
        finally:
            conn.close()
