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


@app.route("/generate", methods=["POST"])  # Original path
@app.route("/api/generate", methods=["POST"])  # Alias for Firebase Hosting rewrite
def generate():
    """
    1. Receive image from frontend
    2. Run wireframe detection on it
    3. Generate HTML/CSS from detected elements
    4. Upload to Firebase Storage
    5. Return file URLs to frontend
    """
    try:
        load_model_if_needed()  # Ensure the model is loaded before using
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
            
            # Ensure the generated_files directory exists
            os.makedirs("generated_files", exist_ok=True)
            
            # Create CSS file with fallback content if the espresso.css template is missing
            css_content = ""
            try:
                main.duplicate_css_file()
                main.generate_css_file(elements, "generated_files/duplicate_espresso.css")
                
                # Read the generated CSS to verify it has content
                if os.path.exists("generated_files/duplicate_espresso.css"):
                    with open("generated_files/duplicate_espresso.css", "r") as f:
                        css_content = f.read()
                    print(f"CSS file generated successfully with {len(css_content)} characters")
                else:
                    print("Warning: CSS file was not created")
                    css_content = "/* Fallback CSS - template file not found */\nbody { background: #fff; }"
            except Exception as e:
                print(f"CSS generation error: {e}")
                css_content = "/* Fallback CSS - generation failed */\nbody { background: #fff; }"

            # Save locally inside container
            html_path = "generated_files/output.html"
            css_path = "generated_files/style.css"

            with open(html_path, "w") as f:
                f.write(html_content)
            
            # Save CSS content directly instead of renaming
            with open(css_path, "w") as f:
                f.write(css_content)

            # Upload to Firebase Storage
            timestamp = int(time.time())
            user_id = request.form.get("user_id", "demoUser")
            html_url = upload_to_storage(html_path, f"generated/{user_id}/{timestamp}/output.html")
            css_url = upload_to_storage(css_path, f"generated/{user_id}/{timestamp}/style.css")

            # Update the HTML file to use the absolute CSS URL
            with open(html_path, "r") as f:
                html_content = f.read()
            
            # Replace the relative CSS link with the absolute URL
            html_content = html_content.replace('href="style.css"', f'href="{css_url}"')
            
            with open(html_path, "w") as f:
                f.write(html_content)

            # Re-upload the updated HTML file with the correct CSS link
            html_url = upload_to_storage(html_path, f"generated/{user_id}/{timestamp}/output.html")

            # Read the final HTML and CSS content to return in the response
            with open(html_path, "r") as f:
                final_html_content = f.read()
            
            with open(css_path, "r") as f:
                final_css_content = f.read()

            # Debug: Print what we're about to return
            print(f"API Response - HTML content length: {len(final_html_content)}")
            print(f"API Response - CSS content length: {len(final_css_content)}")
            print(f"API Response - HTML preview: {final_html_content[:200]}...")
            print(f"API Response - CSS preview: {final_css_content[:200]}...")

            return jsonify({
                "success": True,
                "html_url": html_url,
                "css_url": css_url,
                "html_content": final_html_content,
                "css_content": final_css_content,
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


@app.route("/detect", methods=["POST"])  # Original path
@app.route("/api/detect", methods=["POST"])  # Alias for Firebase Hosting rewrite
def detect_only():
    """
    Endpoint to only run wireframe detection and return JSON
    """
    try:
        load_model_if_needed()  # Ensure the model is loaded before using
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


@app.route("/health", methods=["GET"])  # Original path
@app.route("/api/health", methods=["GET"])  # Alias for Firebase Hosting rewrite
def health():
    """Health check endpoint for Cloud Run"""
    return jsonify({"status": "healthy", "timestamp": time.time()})


@app.route("/", methods=["GET"])
def home():
    return "Image2Web API is running! Endpoints: /generate (POST), /detect (POST), /health (GET)"


model_loaded = False


def load_model_if_needed():
    global model_loaded
    if not model_loaded:
        print("Loading model...")
        try:
            # Try Keras 3 first (for models saved with keras.src.models.functional)
            try:
                import keras
                print(f"Using Keras version: {keras.__version__}")
                visualization.model = keras.models.load_model(
                    visualization.MODEL_PATH,
                    compile=False,
                    safe_mode=False
                )
                print("Model loaded with Keras 3")
            except Exception as keras_err:
                print(f"Keras 3 load failed: {keras_err}")
                # Fallback to tf.keras
                import tensorflow as tf
                print(f"Falling back to TensorFlow version: {tf.__version__}")
                visualization.model = tf.keras.models.load_model(
                    visualization.MODEL_PATH, 
                    compile=False,
                    custom_objects={}
                )
                print("Model loaded with tf.keras")

            model_loaded = True
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise


def create_app():
    """Application factory for better organization"""
    return app


if __name__ == "__main__":
    # Don't load model at startup - load it lazily when needed
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)