#TODO: Fix the Path to referencing the generated CSS files

def generate_just_html (elements):
    """Generate HTML from parsed elements."""
    html_elements = []
    for el in elements:
        #In here We have the elements that will be on the Generated HTML file.
        #We have: Checkbox, Button, Textbox, image, text, navbar, and paragraph.
        tag = None
        if el["label"] == "checkbox":
            tag = f'<input type = "checkbox" style="position:absolute; left:{el["x"]}px; top:{el["y"]}px; width:{el["width"]}px; height:{el["height"]}px;">'
        elif el["label"] == "button":
            tag = f'<button style="position:absolute; left:{el["x"]}px; top:{el["y"]}px; width:{el["width"]}px; height:{el["height"]}px;">Click</button>'
        elif el["label"] == "textbox":
            tag = f'<input type="text" style="position:absolute; left:{el["y"]}px; top:{el["y"]}px; width:{el["width"]}px; height:{el["height"]}px;">'
        elif el["label"] == "image":
            placeholder_url = f"https://via.placeholder.com/{int(el['width'])}x{int(el['height'])}/cccccc/666666?text=Image"
            tag = f'<img src="{placeholder_url}" alt="Image" style="position:absolute; left:{el["x"]}px; top:{el["y"]}px; width:{el["width"]}px; height:{el["height"]}px; object-fit:cover;">'
        #Might want to include heading tags (h1,h2,h3,etc.)
        elif el["label"] == "text":
            tag = f'<span style="position:absolute; left:{el["x"]}px; top:{el["y"]}px; width:{el["width"]}px; height:{el["height"]}px; display:flex; align-items:center;">Sample Text</span>'
        elif el["label"] == "navbar":
            tag = f'<nav style="position:absolute; left:{el["x"]}px; top:{el["y"]}px; width:{el["width"]}px; height:{el["height"]}px; background-color:#f8f9fa; border:1px solid #dee2e6; display:flex; align-items:center; padding:0 15px;"><a href="#" style="margin-right:20px; text-decoration:none; color:#007bff;">Home</a><a href="#" style="margin-right:20px; text-decoration:none; color:#007bff;">About</a><a href="#" style="text-decoration:none; color:#007bff;">Contact</a></nav>'
        elif el["label"] == "paragraph":
            tag = f'<p style="position:absolute; left:{el["x"]}px; top:{el["y"]}px; width:{el["width"]}px; height:{el["height"]}px; margin:0; padding:10px; overflow:hidden;">This is a sample paragraph text that demonstrates how text content would appear in this element.</p>'
        else:
            tag = f'<div style="position:absolute; left:{el["x"]}px; top:{el["y"]}px; width:{el["width"]}px; height:{el["height"]}px; border:1px solid black;">{el["label"]}</div>'
        html_elements.append(tag)
    html_content = f"""<!DOCTYPE html>
<html lang="e">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generated UI</title>
    <link rel="stylesheet" href="resources/espresso.css">
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
        #We have: Checkbox, Button, Textbox, image, text, navbar, and paragraph.
        class_name = f"{el['label']}_{i}"
        tag = None
        if el["label"] == "checkbox":
            tag = f'<input type = "checkbox" class="checkbox-base {class_name}">'
        elif el["label"] == "button":
            tag = f'<button class="button-base {class_name}">Click</button>'
        elif el["label"] == "textbox":
            tag = f'<input type="text" class="textbox-base {class_name}" placeholder="Enter your text here">'
        elif el["label"] == "image":
            placeholder_url = f"https://via.placeholder.com/{int(el['width'])}x{int(el['height'])}/cccccc/666666?text=Image"
            tag = f'<img src="{placeholder_url}" alt="Image" class="image-base {class_name}">'
        #Might want to include heading tags (h1,h2,h3,etc.)
        elif el["label"] == "text":
            tag = f'<span class="text-base {class_name}">Sample Text</span>'
        elif el["label"] == "navbar":
            tag = f'<nav class="navbar-base {class_name}"><a href="#">Home</a><a href="#">About</a><a href="#">Contact</a></nav>'
        elif el["label"] == "paragraph":
            tag = f'<p class="paragraph-base {class_name}">This is a sample paragraph text that demonstrates how text content would appear in this element.</p>'
        else:
            tag = f'<div class="{class_name}">{el["label"]}</div>'
        html_elements.append(tag)
    html_content = f"""<!DOCTYPE html>
<html lang="e">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generated UI</title>
    <link rel="stylesheet" href="duplicate_espresso.css">
</head>
<body>
{chr(10).join(html_elements)}
</body>
</html>"""
    return html_content