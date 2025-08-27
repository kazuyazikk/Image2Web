from flask import Flask, request, jsonify
import main   # The Generator code
from google.cloud import storage
import os, time

app = Flask(__name__)

# configure the firebase storage bucket (same bucket as the Firebase project)
BUCKET_NAME = os.environ.get("BUCKET_NAME", "your-project-id.appspot.com")

def upload_to_storage(local_file, dest_path):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(dest_path)
    blob.upload_from_filename(local_file)
    #blob.make_public()  # optional: makes URL public
    return blob.public_url

@app.route("/generate", methods=["POST"])
def generate():
    """
    1. Receive JSON from frontend (wireframe detection output)
    2. Run main.py generator on it
    3. Save output.html and style.css
    4. Upload to Firebase Storage
    5. Return file URLs to frontend
    """
    json_data = request.get_json()
    if not json_data:
        return jsonify({"error": "No JSON input"}), 400

    #TODO: Include the detection module here

    # Run your existing coding generator
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
    user_id = "demoUser"  # TODO: pass user_id from frontend
    html_url = upload_to_storage(html_path, f"generated/{user_id}/{timestamp}/output.html")
    css_url = upload_to_storage(css_path, f"generated/{user_id}/{timestamp}/style.css")

    return jsonify({
        "html_url": html_url,
        "css_url": css_url
    })

@app.route("/", methods=["GET"])
def home():
    return "HTML/CSS Generator API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
