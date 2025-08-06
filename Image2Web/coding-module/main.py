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
    
def display_elements(data):
    """Display the parsed elements for debugging."""
    if not data:
        return
    
    print(f"Found element: {data['label']}")
    print(f"Shape Type: {data['shape_type']}")
    print(f"Point: {data['points']}")
    print(f"Group ID: {data['group_id']}")
    print(f"-" * 30)
    
if __name__ == "__main__":
    json_data = read_json_file("sample_output.json") #replace with your actual file path
    if json_data:
        display_elements(json_data)