from flask import Flask, request, jsonify
import main

app = Flask(__name__)

@app.route("/generate",methods=["POST"])
def generate():
    #Get JSON input from frontend
    json_data = request.get_json()
    if not json_data:
        return jsonify({"error": "No JSON input"}), 400
    
    #This is from our existing pipeline
    elements = main.parse_elements(json_data)
    html_content = main.generate_html(elements)
    main.duplicate_css_file()
    main.generate_css_file(elements, "generated_files/duplicate_espresso.css")
    
    # Return the results instead of writing only to file
    return jsonify({
        "html": html_content,
        "css_file": "duplicate_espresso.css"
    })
    
@app.route("/", methods=["GET"])
def home():
    return "HTML/CSS Generator API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)