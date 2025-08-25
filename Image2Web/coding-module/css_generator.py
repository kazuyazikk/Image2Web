import shutil
import os

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
                #If you comment the elif condition block of navbar, it makes it an actual element to move around
                #instead of being fixed at the very top of the website
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
                    #f.write(css_rule)
                    continue
                elif el["label"] == "checkbox_with_label":
                    #Just continue
                    css_rule += "}\n\n"
                    f.write(css_rule)
                    continue
                else:
                    css_rule +="  border:1px solid black;\n"
                #Close rule
                css_rule += "}\n\n"
                f.write(css_rule)
        print(f"CSS rules appended successfully -> {css_file}")
    except Exception as e:
        print(f"Error writing CSS: {e}")