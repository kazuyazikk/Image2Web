import json
import shutil
import os
from html_generator import generate_html, generate_just_html

def duplicate_css_file():
    source_file = "resources/espresso.css"
    duplicate_file = "generated_files/duplicate_espresso.css"
    
    try:
        if not os.path.exists(source_file):
            raise FileNotFoundError(f"Source file '{source_file}' does not exist.")
        shutil.copy(source_file, duplicate_file)
        print(f"css File duplicated succesfully! -> {duplicate_file}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

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
    
    #Handle both a single statement dict or a list of elements.
    if isinstance(data, dict):
        data = [data]
    for item in data:
        label = item.get("label", "unknown")
        shape_type = item.get("shape_type", "unknown")
        group_id = item.get("group_id")
        flags = item.get("flags", [])
        points = item.get("points", [])
        unit = item.get("unit")
        
        if(unit is None):
            unit = "px"
        
        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            width = x2 -x1
            height = y2 - y1
            
        
            element_info = {
                "label" : label,
                "shape_type" : shape_type,
                "group_id" : group_id,
                "flags" : flags,
                "x" : x1,
                "y" : y1,
                "width" : width,
                "height" : height,
                "positioning" : "absolute",
                "unit" : unit
            }
            elements.append(element_info)
    return elements
        
def generate_css_file(elements, css_file):
    """Append CSS positioning rules for each element into the duplicated CSS file."""
    try:
        with open(css_file, "a") as f:
            for i, el in enumerate(elements):
                class_name = f"{el['label']}_{i}"
                css_rule = (
                    f".{class_name} {{\n"
                    f"  position: {el['positioning']};\n"
                    f"  left: {el['x']}{el['unit']};\n"
                    f"  top: {el['y']}{el['unit']};\n"
                    f"  width: {el['width']}{el['unit']};\n"
                    f"  height: {el['height']}{el['unit']};\n"
                )
                # Add label-specific extras
                if el["label"] == "image":
                    css_rule += "  object-fit:cover;\n"
                elif el["label"] == "text":
                    css_rule += "  display:flex; align-items:center;\n"
                elif el["label"] == "paragraph":
                    css_rule += "  margin:0; padding:10px; overflow:hidden;\n"
                elif el["label"] == "navbar":
                    css_rule += (
                        "  background-color: #f8f9fa;\n"
                        "  border: 1px solid #dee2e6;\n"
                        "  display: flex;\n"
                        "  align-items: center;\n"
                        "  padding: 0 15px;\n"
                    )
                    css_rule += "}\n"
                    css_rule += (
                        f".{class_name} a {{\n"
                        "  margin-right: 20px;\n"
                        "  text-decoration: none;\n"
                        "  color: #007bff;\n"
                        "}\n\n"
                    )
                    continue
                else:
                    css_rule +="  border:1px solid black;\n"
                #Close rule
                css_rule += "}\n\n"
                f.write(css_rule)
        print(f"CSS rules appended successfully -> {css_file}")
    except Exception as e:
        print(f"Error writing CSS: {e}")

def display_elements(data):
    """Display the parsed elements for debugging."""
    if not data:
        return
    
    print(f"Found element: {data['label']}")
    print(f"Shape Type: {data['shape_type']}")
    print(f"Point: {data['points']}")
    print(f"Group ID: {data['group_id']}")
    print(f"-" * 30) #Line break
    
if __name__ == "__main__":
    json_data = read_json_file("sample_output.json") #replace with your actual file path
    if json_data:
        # display_elements(json_data)
        elements = parse_elements(json_data)
        #html_content = generate_just_html(elements)
        html_content = generate_html(elements)
        
        # Duplicate the css file
        duplicate_css_file()
        
        #Append auto-generated rules
        generate_css_file(elements, "generated_files/duplicate_espresso.css")
        #TODO: Remove the magic variables in the future when we integrate frontend and backend
        #Magic Variables: html_generator, Generating css file, Duplicating css file
        
        #Save to file
        with open("generated_files/output.html", "w") as f:
            f.write(html_content)
        print("HTML file generated!")