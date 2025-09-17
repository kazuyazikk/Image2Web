import shutil
import os

def duplicate_css_file_with_theme(theme_name):
    """
    Duplicate a CSS file from resources based on the selected theme.
    theme_name: 'espresso', 'modern', 'minimal', 'dark'
    """
    theme_file = f"resources/{theme_name}.css"
    duplicate_file = f"generated_files/duplicate_{theme_name}.css"
    try:
        if not os.path.exists(theme_file):
            raise FileNotFoundError(f"Theme file '{theme_file}' does not exist.")
        shutil.copy(theme_file, duplicate_file)
        print(f"CSS file duplicated successfully! -> {duplicate_file}")
        return duplicate_file
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
    return None


def group_elements(elements, threshold=20):
    """Group elements that are spatially close together."""
    groups = []
    visited = set()

    def is_close(el1, el2):
        return not (
            el1['x'] + el1['width'] + threshold < el2['x'] or
            el2['x'] + el2['width'] + threshold < el1['x'] or
            el1['y'] + el1['height'] + threshold < el2['y'] or
            el2['y'] + el2['height'] + threshold < el1['y']
        )

    for i, el in enumerate(elements):
        if i in visited:
            continue
        group = [el]
        visited.add(i)
        for j, other in enumerate(elements):
            if j not in visited and is_close(el, other):
                group.append(other)
                visited.add(j)
        groups.append(group)
    return groups


def generate_css_file(elements, css_file):
    """Append CSS positioning rules for each element into the duplicated CSS file, grouped into divs."""
    try:
        groups = group_elements(elements)
        with open(css_file, "a") as f:
            # Write CSS for groups
            for g_idx, group in enumerate(groups):
                css_rule = (
                    f".group_{g_idx} {{\n"
                    f"  display: flex;\n"
                    f"  gap: 10px;\n"
                    f"  margin-bottom: 20px;\n"
                    f"}}\n\n"
                )
                f.write(css_rule)

                for i, el in enumerate(group):
                    class_name = f"{el['label']}_{i}_g{g_idx}"
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
                    # elif el["label"] == "navbar":
                    #     css_rule += (
                    #         "  background-color: #f8f9fa;\n"
                    #         "  border: 1px solid #dee2e6;\n"
                    #         "  display: flex;\n"
                    #         "  align-items: center;\n"
                    #         "  padding: 0 15px;\n"
                    #     )
                    #     css_rule += "}\n"
                    #     css_rule += (
                    #         f".{class_name} a {{\n"
                    #         "  margin-right: 20px;\n"
                    #         "  text-decoration: none;\n"
                    #         "  color: #007bff;\n"
                    #         "}}\n\n"
                    #     )
                    #     f.write(css_rule)
                    #     continue
                    elif el["label"] == "checkbox_with_label":
                        css_rule += "}\n\n"
                        f.write(css_rule)
                        continue
                    else:
                        css_rule += "  border:1px solid black;\n"

                    css_rule += "}\n\n"
                    f.write(css_rule)
        print(f"CSS rules appended successfully with grouping -> {css_file}")
    except Exception as e:
        print(f"Error writing CSS: {e}")
