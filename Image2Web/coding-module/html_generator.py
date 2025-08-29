def generate_just_html (elements):
    """Generate HTML from parsed elements."""
    html_elements = []
    for el in elements:
        #In here We have the elements that will be on the Generated HTML file.
        #We have: Checkbox, Button, Textfield, image, text, navbar, and paragraph.
        tag = None
        if el["label"] == "checkbox":
            tag = f'<input type = "checkbox" style="position:absolute; left:{el["x"]}px; top:{el["y"]}px; width:{el["width"]}px; height:{el["height"]}px;">'
        elif el["label"] == "button":
            tag = f'<button style="position:absolute; left:{el["x"]}px; top:{el["y"]}px; width:{el["width"]}px; height:{el["height"]}px;">{label_description(el["label"])}</button>'
        elif el["label"] == "textfield":
            tag = f'<input type="text" style="position:absolute; left:{el["y"]}px; top:{el["y"]}px; width:{el["width"]}px; height:{el["height"]}px;" {label_description(el["label"])}>'
        elif el["label"] == "image":
            placeholder_url = f"https://via.placeholder.com/{int(el['width'])}x{int(el['height'])}/cccccc/666666?text=Image"
            tag = f'<img src="{placeholder_url}" alt="Image" style="position:absolute; left:{el["x"]}px; top:{el["y"]}px; width:{el["width"]}px; height:{el["height"]}px; object-fit:cover;">'
        #Might want to include heading tags (h1,h2,h3,etc.)
        elif el["label"] == "text":
            tag = f'<span style="position:absolute; left:{el["x"]}px; top:{el["y"]}px; width:{el["width"]}px; height:{el["height"]}px; display:flex; align-items:center;"> {label_description(el["label"])} </span>'
        elif el["label"] == "navbar":
            tag = f'<nav style="position:absolute; left:{el["x"]}px; top:{el["y"]}px; width:{el["width"]}px; height:{el["height"]}px; background-color:#f8f9fa; border:1px solid #dee2e6; display:flex; align-items:center; padding:0 15px;"><a href="#" style="margin-right:20px; text-decoration:none; color:#007bff;">Home</a><a href="#" style="margin-right:20px; text-decoration:none; color:#007bff;">About</a><a href="#" style="text-decoration:none; color:#007bff;">Contact</a></nav>'
        elif el["label"] == "paragraph":
            tag = f'<p style="position:absolute; left:{el["x"]}px; top:{el["y"]}px; width:{el["width"]}px; height:{el["height"]}px; margin:0; padding:10px; overflow:hidden;">{label_description(el["label"])}</p>'
        else:
            tag = f'<div style="position:absolute; left:{el["x"]}px; top:{el["y"]}px; width:{el["width"]}px; height:{el["height"]}px; border:1px solid black;">{el["label"]}</div>'
        html_elements.append(tag)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generated UI</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
{chr(10).join(html_elements)}
</body>
</html>"""
    return html_content

def generate_html (elements):
    """Generate HTML from parsed elements. But it has integration with css"""
    html_elements = []
    for i, el in enumerate(elements):
        #In here We have the elements that will be on the Generated HTML file.
        #We have: Checkbox, Button, Textfield, image, text, navbar, and paragraph.
        class_name = f"{el['label']}_{i}"
        tag = None
        if el["label"] == "checkbox":
            tag = f'<input type = "checkbox" class="checkbox-base {class_name}">'
        elif el["label"] == "button":
            tag = f'<button class="button-base {class_name}">{label_description(el["label"])}</button>'
        elif el["label"] == "textfield":
            tag = f'<input type="text" class="textfield-base {class_name}" {label_description(el["label"])}>'
        elif el["label"] == "image":
            placeholder_url = f"https://via.placeholder.com/{int(el['width'])}x{int(el['height'])}/cccccc/666666?text=Image"
            tag = f'<img src="{placeholder_url}" alt="Image" class="image-base {class_name}">'
        #Might want to include heading tags (h1,h2,h3,etc.)
        elif el["label"] == "text":
            tag = f'<span class="text-base {class_name}">{label_description(el["label"])}</span>'
        elif el["label"] == "navbar":
            tag = f'<nav class="navbar-base {class_name}"><a href="#">Home</a><a href="#">About</a><a href="#">Contact</a></nav>'
        elif el["label"] == "paragraph":
            tag = f'<p class="paragraph-base {class_name}">{label_description(el["label"])}</p>'
        elif el["label"] == "checkbox_with_label":
            tag = f"""<label class="checkbox-label-base {class_name}">
            <input type="checkbox"> This is a checkbox and a label
            </label>"""
        else:
            tag = f'<div class="{class_name}">{el["label"]}</div>'
        html_elements.append(tag)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generated UI</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
{chr(10).join(html_elements)}
</body>
</html>"""
    return html_content

#Elements and their descriptions
#For easy changing of description if ever the need arises.
def label_description(label):
    if label == "button":
        return("Click")
    if label == "textfield":
        return("placeholder='Enter your text here'")
    if label == "text":
        return("This is a Sample text, different from paragraph.")
    if label == "paragraph":
        return("This is a sample paragraph text that demonstrates how text content would appear in this element.")