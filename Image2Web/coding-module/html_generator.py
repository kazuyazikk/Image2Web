def generate_html(elements):
    """Generate HTML with grouped elements based on alignment_type."""
    html_elements = []
    already_have_navbar = False

    # Group by alignment_type
    grouped = {}
    for el in elements:
        gtype = el.get("alignment_type", "single")
        grouped.setdefault(gtype, []).append(el)

    for g_idx, (atype, group) in enumerate(grouped.items()):
        #open the group heading
        group_tags = [f'<div class="group_{g_idx} {atype}">']
        for i, el in enumerate(group):
            class_name = f"{el['label']}_{i}_g{g_idx}"
            tag = None

            if el["label"] == "checkbox":
                tag = f'<input type="checkbox" class="checkbox-base {class_name}">' 
            elif el["label"] == "button":
                tag = f'<button class="button-base {class_name}">{label_description(el["label"])}</button>'
            elif el["label"] == "textfield":
                tag = f'<input type="text" class="textfield-base {class_name}" {label_description(el["label"])}>'
            elif el["label"] == "image":
                placeholder_url = f"https://via.placeholder.com/{int(el['width'])}x{int(el['height'])}/cccccc/666666?text=Image"
                tag = f'<img src="{placeholder_url}" alt="Image" class="image-base {class_name}">' 
            elif el["label"] == "text":
                tag = f'<h6 class="text-base {class_name}">{label_description(el["label"])}</h6>'
            elif el["label"] == "navbar" and not already_have_navbar:
                nav_class_name = f"{el['label']}_{i}"
                tag = f'''
                <nav class="navbar-base">
                    <div class="nav-content">
                        <ul class="nav-links">
                            <li><a href="#">Home</a></li>
                            <li><a href="#">About</a></li>
                            <li><a href="#">Contact</a></li>
                        </ul>
                        <input type="text" placeholder="Search..." class="navbar-search">
                    </div>
                </nav>'''
                html_elements.append(tag)
                already_have_navbar = True
                continue
            elif el["label"] == "paragraph":
                tag = f'<p class="paragraph-base {class_name}">{label_description(el["label"])}</p>'
            elif el["label"] == "checkbox_with_label":
                tag = f'<label class="checkbox-label-base {class_name}"><input type="checkbox"> This is a checkbox and a label</label>'
            else:
                tag = f'<div class="{class_name}">{el["label"]}</div>'

            group_tags.append(tag)

        #close the heading
        group_tags.append("</div>")
        html_elements.append("\n".join(group_tags))

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Generated UI</title>
    <link rel=\"stylesheet\" href=\"style.css\">
</head>
<body>
{chr(10).join(html_elements)}
</body>
</html>"""


#Elements and their descriptions
#For easy changing of description if ever the need arises.
def label_description(label):
    if label == "button":
        return("Click")
    if label == "textfield":
        return("placeholder='Enter your text here'")
    if label == "text":
        return("This is a Sample text!")
    if label == "paragraph":
        return("This is a sample paragraph text!")