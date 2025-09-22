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

    # Case 1: flat list of shapes with "points"
    if isinstance(data, list) and data and "points" in data[0]:
        for idx, el in enumerate(data):
            (x1, y1), (x2, y2) = el["points"]
            width, height = x2 - x1, y2 - y1

            element_info = {
                "label": el.get("label", "unknown"),
                "group_index": el.get("group_id", idx),
                "x": x1,
                "y": y1,
                "width": width,
                "height": height,
                "positioning": "absolute",
                "unit": el.get("unit", "%"),
                "group_type": "single_element",
                "alignment_type": "single"
            }
            elements.append(element_info)

    # Case 2: grouped JSON with "groups"
    elif isinstance(data, dict) and "groups" in data:
        for g_idx, group in enumerate(data["groups"]):
            for el in group["elements"]:
                bbox = el.get("bounding_box", {})
                x1, y1, x2, y2 = bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)
                width, height = x2 - x1, y2 - y1

                element_info = {
                    "label": el.get("label", "unknown"),
                    "group_index": g_idx,
                    "x": x1,
                    "y": y1,
                    "width": width,
                    "height": height,
                    "positioning": "absolute",
                    "unit": "%",
                    "group_type": group.get("type"),
                    "alignment_type": group.get("alignment_type", "single")
                }
                elements.append(element_info)

    else:
        print("parse_elements: Unrecognized JSON structure", type(data))
        return []

    print(f"parse_elements returned: {type(elements)} with {len(elements)} elements")
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