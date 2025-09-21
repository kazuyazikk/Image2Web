import json
from html_generator import generate_html
from css_generator import duplicate_css_file_with_theme, generate_css_file

def read_json_file(file_path):
    """Reads and parse the JSON file from the CNN module."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        print("JSON file loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON file.")
        return None
    
def parse_elements(data):
    """Parse JSON elements into a structured format with positioning info."""
    elements = []
    
    if isinstance(data, dict) and "groups" in data:
        for g_idx, group in enumerate(data["groups"]):
            for el in group["elements"]:
                bbox = el.get("bounding_box",{})
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                width, height = x2 - x1, y2 - y1
                
                element_info = {
                    "label": el.get("label", "unknown"),
                    "x": x1,
                    "y": y1,
                    "width": width,
                    "height": height,
                    "positioning": "absolute",
                    "unit": "rem",
                    "group_type": group.get("type"),
                    "alignment_type": group.get("alignment_type")
                }
                elements.append(element_info)
        return elements
    
    
if __name__ == "__main__":
    json_data = read_json_file("sample_output.json") 
    if json_data:
        elements = parse_elements(json_data)
        html_content = generate_html(elements)
        
        # Duplicate the css file
        duplicate_css_file_with_theme('espresso')
        
        #Append auto-generated rules
        generate_css_file(elements, "generated_files/duplicate_espresso.css")
        
        #Save to file
        with open("generated_files/output.html", "w") as f:
            f.write(html_content)
        print("HTML file generated!")