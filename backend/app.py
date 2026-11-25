
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
from werkzeug.utils import secure_filename
import traceback
import sys

# --------------------------------------------------
# Import PatchCore inspection service
# --------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent

# Add backend folder to Python import path
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ai.patchcore_service import inspect_image


# --------------------------------------------------
# Flask application setup
# --------------------------------------------------

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = PROJECT_DIR / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}


# --------------------------------------------------
# File validation
# --------------------------------------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# --------------------------------------------------
# Home route
# --------------------------------------------------

@app.route("/")
def home():
    return "PROMETHEUS-X Backend is running!"


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.route("/health")
def health():
    return jsonify({
        "status": "success",
        "message": "Backend is working"
    })


# --------------------------------------------------
# AI image inspection route
# --------------------------------------------------

@app.route("/inspect", methods=["POST"])
def inspect():
    try:
        # Check uploaded image
        if "image" not in request.files:
            return jsonify({
                "status": "error",
                "message": "Upload an image using the field name 'image'"
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "status": "error",
                "message": "No image selected"
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "status": "error",
                "message": "Only PNG, JPG, JPEG and BMP images are allowed"
            }), 400

        # Save uploaded image
        filename = secure_filename(file.filename)
        image_path = UPLOAD_FOLDER / filename

        file.save(image_path)

        # Run PatchCore inspection
        result = inspect_image(str(image_path))

        # Return result to frontend
        return jsonify({
            "status": "success",
            "message": "AI inspection completed",
            "result": result
        })

    except Exception as error:
        traceback.print_exc()

        return jsonify({
            "status": "error",
            "message": str(error)
        }), 500


# --------------------------------------------------
# Run backend
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
