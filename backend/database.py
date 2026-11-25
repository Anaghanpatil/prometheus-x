import sqlite3
from datetime import datetime

DATABASE_PATH = "inspection_memory.db"


def create_database():
    connection = sqlite3.connect(DATABASE_PATH)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            result TEXT,
            anomaly_score REAL,
            severity TEXT,
            inspected_at TEXT
        )
    """)

    connection.commit()
    connection.close()


def save_inspection(image_name, result, anomaly_score, severity):
    connection = sqlite3.connect(DATABASE_PATH)

    connection.execute("""
        INSERT INTO inspections
        (image_name, result, anomaly_score, severity, inspected_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        image_name,
        result,
        anomaly_score,
        severity,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    connection.commit()
    connection.close()


def get_inspection_history():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    records = connection.execute("""
        SELECT * FROM inspections
        ORDER BY id DESC
    """).fetchall()

    connection.close()

    return [dict(record) for record in records]