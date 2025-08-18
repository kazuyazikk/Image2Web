import json

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
        points = item.get(points, [])
        
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
                "positioning" : "absolute"
            }
            elements.append(element_info)
    return elements
        
def generate_html (elements):
    """Generate HTML from parsed elements."""
    html_elements = []
    for el in elements:
        tag = None
        if el["label"] == "checkbox":
            tag = f'<input type = "checkbox" style="position:absolute; left:{el["x"]}px; top:{el["y"]}px; width:{el["width"]}px; height:{el["height"]}px;">'
        elif el["label"] == "button":
            tag = f'<button style="position:absolute; left:{el["x"]}px; top:{el["y"]}px; width:{el["width"]}px; height:{el["height"]}px;">Click</button>'
        elif el["label"] == "textbox":
            tag = f'<input type="text" style="position:absolute; left:{el["y"]}px; top:{el["y"]}px; width:{el["width"]}px; height:{el["height"]}px;">'
        #TODO: Put in the remaining Elements from the HTML here
        else:
            tag = f'<div style="position:absolute; left:{el["x"]}px; top:{el["y"]}px; width:{el["width"]}px; height:{el["height"]}px; border:1px solid black;">{el["label"]}</div>'
        html_elements.append(tag)
    html_content = "<!DOCTYPE html>\n<html>\n<body>\n" + "\n".join(html_elements) + "\n</body>\n</html>"
    return html_content


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
        html_content = generate_html(elements)
        
        #Save to file
        with open("output.html", "w") as f:
            f.write(html_content)
        print("HTML file generated!")