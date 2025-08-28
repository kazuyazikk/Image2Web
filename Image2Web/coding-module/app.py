from flask import Flask, request, jsonify
import main
import visualization
from google.cloud import storage
import os, time
import tempfile
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure the firebase storage bucket (same bucket as the Firebase project)
BUCKET_NAME = os.environ.get("BUCKET_NAME", "your-project-id.appspot.com")


def upload_to_storage(local_file, dest_path):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(dest_path)
    blob.upload_from_filename(local_file)
    return blob.public_url


@app.route("/generate", methods=["POST"])
def generate():
    """
    1. Receive image from frontend
    2. Run wireframe detection on it
    3. Generate HTML/CSS from detected elements
    4. Upload to Firebase Storage
    5. Return file URLs to frontend
    """
    try:
        load_model_if_needed() #Ensure the model is loaded before using
        # Check if an image file was uploaded
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            file.save(temp_file.name)
            temp_image_path = temp_file.name

        try:
            # Run wireframe detection using your visualization module
            print("Running wireframe detection...")
            pil_img, final_boxes = visualization.get_predictions(temp_image_path)

            # Convert predictions to JSON format using your existing function
            json_data = []
            if final_boxes:
                # Create a temporary file for JSON output
                temp_json_path = temp_image_path.replace(os.path.splitext(temp_image_path)[1], '_pred.json')
                visualization.predictions_to_json(final_boxes, temp_json_path, pil_img.width, pil_img.height)

                # Read the JSON file
                with open(temp_json_path, 'r') as f:
                    json_data = json.load(f)

                # Clean up temp JSON file
                os.unlink(temp_json_path)

            print(f"Detected {len(json_data)} elements")

            # Run your existing code generator
            elements = main.parse_elements(json_data)
            html_content = main.generate_html(elements)
            main.duplicate_css_file()
            main.generate_css_file(elements, "generated_files/duplicate_espresso.css")

            # Save locally inside container
            os.makedirs("generated_files", exist_ok=True)
            html_path = "generated_files/output.html"
            css_path = "generated_files/style.css"

            with open(html_path, "w") as f:
                f.write(html_content)
            os.rename("generated_files/duplicate_espresso.css", css_path)

            # Upload to Firebase Storage
            timestamp = int(time.time())
            user_id = request.form.get("user_id", "demoUser")
            html_url = upload_to_storage(html_path, f"generated/{user_id}/{timestamp}/output.html")
            css_url = upload_to_storage(css_path, f"generated/{user_id}/{timestamp}/style.css")

            return jsonify({
                "success": True,
                "html_url": html_url,
                "css_url": css_url,
                "detected_elements": len(json_data),
                "elements": json_data  # Include detected elements for debugging
            })

        finally:
            # Clean up temporary file
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)

    except Exception as e:
        print(f"Error in generate endpoint: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route("/detect", methods=["POST"])
def detect_only():
    """
    Endpoint to only run wireframe detection and return JSON
    """
    try:
        load_model_if_needed() # Ensure the model is loaded before using
        # Check if an image file was uploaded
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            file.save(temp_file.name)
            temp_image_path = temp_file.name

        try:
            # Run wireframe detection using your visualization module
            print("Running wireframe detection...")
            pil_img, final_boxes = visualization.get_predictions(temp_image_path)

            # Convert predictions to JSON format
            json_data = []
            if final_boxes:
                # Create a temporary file for JSON output
                temp_json_path = temp_image_path.replace(os.path.splitext(temp_image_path)[1], '_pred.json')
                visualization.predictions_to_json(final_boxes, temp_json_path, pil_img.width, pil_img.height)

                # Read the JSON file
                with open(temp_json_path, 'r') as f:
                    json_data = json.load(f)

                # Clean up temp JSON file
                os.unlink(temp_json_path)

            return jsonify({
                "success": True,
                "detected_elements": len(json_data),
                "elements": json_data,
                "image_dimensions": {
                    "width": pil_img.width,
                    "height": pil_img.height
                }
            })

        finally:
            # Clean up temporary file
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)

    except Exception as e:
        print(f"Error in detect endpoint: {str(e)}")
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500


@app.route("/", methods=["GET"])
def home():
    return "Image2Web API is running! Endpoints: /generate (POST), /detect (POST)"

model_loaded = False

def load_model_if_needed():
    global model_loaded
    if not model_loaded:
        print("Loading model...")
        visualization.model = visualization.tf.keras.models.load_model(
            visualization.MODEL_PATH, compile=False
        )
        model_loaded = True
        print("Model Loaded successfully!")

# Load model when the app starts
@app.before_first_request
def initialize():
    # Don't preload the model here anymore
    pass

if __name__ == "__main__":
    # Load model for local testing
    load_model_if_needed()
    app.run(host="0.0.0.0", port=8080, debug=True)