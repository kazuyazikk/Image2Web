import tensorflow as tf
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageOps
import json
import os
import math

model = None  # Makes model global variable

# CONFIG
MODEL_PATH = "./wireframe_detection_model_best.keras"
OUTPUT_DIR = "./output/"
CLASS_NAMES = ["button", "checkbox", "image", "navbar", "paragraph", "text", "textfield"]

IMG_SIZE = 416
GRID_SIZE = 16
CONF_THRESHOLD = 0.1
IOU_THRESHOLD = 0.1

# Element grouping parameters
HORIZONTAL_GROUP_THRESHOLD = 50  # Maximum horizontal distance to group elements
VERTICAL_GROUP_THRESHOLD = 2500  # Maximum vertical distance to group elements
ALIGNMENT_TOLERANCE = 30  # Tolerance for alignment checks


# Checkbox-Text Association Logic
def is_text_next_to_checkbox(checkbox_box, text_box, max_distance=150, alignment_tolerance=50):
    """Check if text element is positioned next to a checkbox with adjustable sensitivity."""
    cb_cls, cb_score, cb_x1, cb_y1, cb_x2, cb_y2 = checkbox_box
    text_cls, text_score, text_x1, text_y1, text_x2, text_y2 = text_box

    # Calculate centers and dimensions
    cb_center_x = (cb_x1 + cb_x2) / 2
    cb_center_y = (cb_y1 + cb_y2) / 2
    cb_width = cb_x2 - cb_x1
    cb_height = cb_y2 - cb_y1

    text_center_x = (text_x1 + text_x2) / 2
    text_center_y = (text_y1 + text_y2) / 2
    text_width = text_x2 - text_x1
    text_height = text_y2 - text_y1

    # Horizontal alignment check
    horizontal_aligned = (
            abs(cb_center_y - text_center_y) <= max(cb_height / 2 + text_height / 2,
                                                    alignment_tolerance) and
            text_x1 >= cb_x1 - 20 and  # Allow some overlap/gap
            text_x1 - cb_x2 <= max_distance  # Distance threshold
    )

    # Vertical alignment check
    vertical_aligned = (
            abs(cb_center_x - text_center_x) <= max(cb_width / 2 + text_width / 2,
                                                    alignment_tolerance) and
            text_y1 >= cb_y1 - 20 and  # Allow some overlap/gap
            text_y1 - cb_y2 <= max_distance  # Distance threshold
    )

    # Additional check: text to the left of checkbox
    left_aligned = (
            abs(cb_center_y - text_center_y) <= max(cb_height / 2 + text_height / 2, alignment_tolerance) and
            text_x2 <= cb_x2 + 20 and  # Text ends before or slightly after checkbox starts
            cb_x1 - text_x2 <= max_distance  # Distance threshold
    )

    # Additional check: text above checkbox
    above_aligned = (
            abs(cb_center_x - text_center_x) <= max(cb_width / 2 + text_width / 2, alignment_tolerance) and
            text_y2 <= cb_y2 + 20 and  # Text ends before or slightly after checkbox starts
            cb_y1 - text_y2 <= max_distance  # Distance threshold
    )

    return horizontal_aligned or vertical_aligned or left_aligned or above_aligned


def are_elements_alignable(box1, box2, max_horizontal_gap=HORIZONTAL_GROUP_THRESHOLD,
                           max_vertical_gap=VERTICAL_GROUP_THRESHOLD,
                           alignment_tolerance=ALIGNMENT_TOLERANCE):
    """Check if two elements should be grouped together for better alignment."""
    cls1, score1, x1_1, y1_1, x2_1, y2_1 = box1
    cls2, score2, x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate centers and dimensions
    center1_x = (x1_1 + x2_1) / 2
    center1_y = (y1_1 + y2_1) / 2
    width1 = x2_1 - x1_1
    height1 = y2_1 - y1_1

    center2_x = (x1_2 + x2_2) / 2
    center2_y = (y1_2 + y2_2) / 2
    width2 = x2_2 - x1_2
    height2 = y2_2 - y1_2

    # Calculate distances
    horizontal_distance = min(
        abs(x1_1 - x2_2),  # box1 left edge to box2 right edge
        abs(x2_1 - x1_2),  # box1 right edge to box2 left edge
        abs(x1_1 - x1_2),  # left edges
        abs(x2_1 - x2_2)  # right edges
    )

    vertical_distance = min(
        abs(y1_1 - y2_2),  # box1 top edge to box2 bottom edge
        abs(y2_1 - y1_2),  # box1 bottom edge to box2 top edge
        abs(y1_1 - y1_2),  # top edges
        abs(y2_1 - y2_2)  # bottom edges
    )

    # Check for horizontal alignment (same row)
    horizontal_aligned = (
            abs(center1_y - center2_y) <= alignment_tolerance and
            horizontal_distance <= max_horizontal_gap
    )

    # Check for vertical alignment (same column)
    vertical_aligned = (
            abs(center1_x - center2_x) <= alignment_tolerance and
            vertical_distance <= max_vertical_gap
    )

    # Check if elements are close enough to be in a form group
    form_group_aligned = (
            horizontal_distance <= max_horizontal_gap * 1.5 and
            vertical_distance <= max_vertical_gap * 1.5
    )

    return {
        'alignable': horizontal_aligned or vertical_aligned or form_group_aligned,
        'alignment_type': 'horizontal' if horizontal_aligned else ('vertical' if vertical_aligned else 'form_group'),
        'horizontal_distance': horizontal_distance,
        'vertical_distance': vertical_distance
    }


def find_element_groups(final_boxes):
    """Group nearby elements that should be aligned together."""
    groups = []
    used_indices = set()

    # Sort boxes by y-coordinate (top to bottom) for consistent grouping
    sorted_boxes = sorted(enumerate(final_boxes), key=lambda x: x[1][3])  # Sort by y1

    for i, (original_idx, box) in enumerate(sorted_boxes):
        if original_idx in used_indices:
            continue

        current_group = {
            'elements': [{'index': original_idx, 'box': box}],
            'alignment_type': 'single',
            'bounding_box': list(box[2:6])  # [x1, y1, x2, y2]
        }

        # Look for nearby elements to group with
        for j, (other_idx, other_box) in enumerate(sorted_boxes):
            if other_idx == original_idx or other_idx in used_indices:
                continue

            alignment_info = are_elements_alignable(box, other_box)

            if alignment_info['alignable']:
                current_group['elements'].append({'index': other_idx, 'box': other_box})
                current_group['alignment_type'] = alignment_info['alignment_type']

                # Update bounding box to encompass all elements
                other_x1, other_y1, other_x2, other_y2 = other_box[2:6]
                current_group['bounding_box'][0] = min(current_group['bounding_box'][0], other_x1)
                current_group['bounding_box'][1] = min(current_group['bounding_box'][1], other_y1)
                current_group['bounding_box'][2] = max(current_group['bounding_box'][2], other_x2)
                current_group['bounding_box'][3] = max(current_group['bounding_box'][3], other_y2)

                used_indices.add(other_idx)

        used_indices.add(original_idx)
        groups.append(current_group)

    return groups


def group_checkbox_with_text(final_boxes):  # Groups checkboxes with nearby text
    grouped_boxes = []
    used_indices = set()

    for i, box in enumerate(final_boxes):
        if i in used_indices:
            continue

        cls, score, x1, y1, x2, y2 = box

        if cls == "checkbox":
            # Look for nearby text elements
            best_text = None
            best_text_idx = None
            best_distance = float('inf')

            for j, other_box in enumerate(final_boxes):
                if j != i and j not in used_indices:
                    other_cls = other_box[0]
                    if other_cls == "text":
                        if is_text_next_to_checkbox(box, other_box):
                            # Calculate distance to find the closest text
                            cb_center_x = (x1 + x2) / 2
                            cb_center_y = (y1 + y2) / 2
                            text_center_x = (other_box[2] + other_box[4]) / 2
                            text_center_y = (other_box[3] + other_box[5]) / 2
                            distance = math.sqrt(
                                (cb_center_x - text_center_x) ** 2 + (cb_center_y - text_center_y) ** 2)

                            if distance < best_distance:
                                best_distance = distance
                                best_text = other_box
                                best_text_idx = j

            if best_text:
                # Create grouped element
                text_cls, text_score, text_x1, text_y1, text_x2, text_y2 = best_text

                # Determine layout type
                layout = "horizontal" if text_x1 >= x2 - 10 else "vertical"

                # Create bounding box that encompasses both elements
                combined_x1 = min(x1, text_x1)
                combined_y1 = min(y1, text_y1)
                combined_x2 = max(x2, text_x2)
                combined_y2 = max(y2, text_y2)

                # Use the higher confidence score
                combined_score = max(score, text_score)

                grouped_boxes.append({
                    "type": "checkbox_with_label",
                    "label": "checkbox_with_label",
                    "score": combined_score,
                    "layout": layout,
                    "checkbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": score},
                    "text": {"x1": text_x1, "y1": text_y1, "x2": text_x2, "y2": text_y2, "score": text_score},
                    "combined_box": [combined_x1, combined_y1, combined_x2, combined_y2]
                })
                used_indices.add(i)
                used_indices.add(best_text_idx)
            else:
                # Standalone checkbox
                grouped_boxes.append({
                    "type": "standalone",
                    "label": cls,
                    "score": score,
                    "box": [x1, y1, x2, y2]
                })
                used_indices.add(i)
        else:
            # Non-checkbox element
            grouped_boxes.append({
                "type": "standalone",
                "label": cls,
                "score": score,
                "box": [x1, y1, x2, y2]
            })
            used_indices.add(i)

    return grouped_boxes


def group_elements_for_layout(final_boxes):
    """Enhanced grouping that combines checkbox-text pairing with general element grouping."""
    # First, handle checkbox-text pairing
    checkbox_text_groups = group_checkbox_with_text(final_boxes)

    # Convert to a format compatible with general grouping
    processed_boxes = []
    for group in checkbox_text_groups:
        if group["type"] == "checkbox_with_label":
            # Use the combined bounding box for further grouping
            box_data = (group["label"], group["score"]) + tuple(group["combined_box"])
            processed_boxes.append(box_data)
        else:
            # Standalone element
            box_data = (group["label"], group["score"]) + tuple(group["box"])
            processed_boxes.append(box_data)

    # Now apply general element grouping
    element_groups = find_element_groups(processed_boxes)

    # Create final grouped structure
    final_groups = []

    for group in element_groups:
        if len(group['elements']) == 1:
            # Single element group
            element = group['elements'][0]
            box_data = element['box']

            # Check if this was originally a checkbox_with_label
            original_group = next((g for g in checkbox_text_groups if
                                   g.get("combined_box") == list(box_data[2:6]) or
                                   g.get("box") == list(box_data[2:6])), None)

            if original_group and original_group["type"] == "checkbox_with_label":
                final_groups.append({
                    "type": "single_checkbox_with_label",
                    "elements": [original_group],
                    "alignment_type": "single",
                    "bounding_box": original_group["combined_box"]
                })
            else:
                final_groups.append({
                    "type": "single_element",
                    "elements": [{
                        "type": "standalone",
                        "label": box_data[0],
                        "score": box_data[1],
                        "box": list(box_data[2:6])
                    }],
                    "alignment_type": "single",
                    "bounding_box": list(box_data[2:6])
                })
        else:
            # Multi-element group
            group_elements = []
            for element in group['elements']:
                box_data = element['box']

                # Check if this was originally a checkbox_with_label
                original_group = next((g for g in checkbox_text_groups if
                                       g.get("combined_box") == list(box_data[2:6]) or
                                       g.get("box") == list(box_data[2:6])), None)

                if original_group and original_group["type"] == "checkbox_with_label":
                    group_elements.append(original_group)
                else:
                    group_elements.append({
                        "type": "standalone",
                        "label": box_data[0],
                        "score": box_data[1],
                        "box": list(box_data[2:6])
                    })

            final_groups.append({
                "type": "element_group",
                "elements": group_elements,
                "alignment_type": group['alignment_type'],
                "bounding_box": group['bounding_box']
            })

    return final_groups


# Get predictions from model, with NMS
def get_predictions(image_path):
    global model
    if model is None:
        raise ValueError("Model not loaded. Please load the model first.")
    pil_img = Image.open(image_path).convert("RGB")
    pil_img = ImageOps.exif_transpose(pil_img)
    orig_w, orig_h = pil_img.size
    resized_img = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img_array = np.array(resized_img, dtype=np.float32) / 255.0
    input_tensor = np.expand_dims(img_array, axis=0)

    # Run prediction
    pred_grid = model.predict(input_tensor, verbose=0)[0]
    raw_boxes = []
    S = pred_grid.shape[0]
    cell_size = 1.0 / S

    for row in range(S):
        for col in range(S):
            obj_score = float(tf.nn.sigmoid(pred_grid[row, col, 0]))
            if obj_score < CONF_THRESHOLD:
                continue

            # Box parameters
            x_offset = float(tf.nn.sigmoid(pred_grid[row, col, 1]))
            y_offset = float(tf.nn.sigmoid(pred_grid[row, col, 2]))
            width = float(tf.nn.sigmoid(pred_grid[row, col, 3]))
            height = float(tf.nn.sigmoid(pred_grid[row, col, 4]))

            # Class info
            class_logits = pred_grid[row, col, 5:]
            class_probs = tf.nn.softmax(class_logits).numpy()
            class_id = int(np.argmax(class_probs))
            class_conf = float(class_probs[class_id])
            final_score = obj_score * class_conf
            if final_score < CONF_THRESHOLD:
                continue

            # Convert to absolute coords
            center_x = (col + x_offset) * cell_size
            center_y = (row + y_offset) * cell_size
            x1 = (center_x - width / 2) * orig_w
            y1 = (center_y - height / 2) * orig_h
            x2 = (center_x + width / 2) * orig_w
            y2 = (center_y + height / 2) * orig_h

            if x2 > x1 and y2 > y1:
                raw_boxes.append((class_id, final_score, x1, y1, x2, y2))

    # Apply NMS per class
    final_boxes = []
    for class_id in range(len(CLASS_NAMES)):
        class_boxes = [(score, x1, y1, x2, y2) for cid, score, x1, y1, x2, y2 in raw_boxes if cid == class_id]
        if not class_boxes:
            continue

        scores = [b[0] for b in class_boxes]
        boxes_xyxy = [[b[1], b[2], b[3], b[4]] for b in class_boxes]  # [x1,y1,x2,y2]

        selected_indices = tf.image.non_max_suppression(
            boxes=boxes_xyxy,
            scores=scores,
            max_output_size=50,
            iou_threshold=IOU_THRESHOLD,
            score_threshold=CONF_THRESHOLD
        )

        for idx in selected_indices.numpy():
            score, x1, y1, x2, y2 = class_boxes[idx]
            final_boxes.append((CLASS_NAMES[class_id], score, x1, y1, x2, y2))

    return pil_img, final_boxes


# Outputs predictions to JSON files
def predictions_to_json(data, save_path, orig_w, orig_h):
    """Save predictions to JSON - handles both old final_boxes format and new grouped format."""

    # Check if we're dealing with old format (list of tuples) or new format (list of dicts)
    if data and isinstance(data[0], tuple):
        # Old format: list of (class, score, x1, y1, x2, y2) tuples
        shapes = []
        for cls, score, x1, y1, x2, y2 in data:
            # Scale coords to percentages
            x1_scaled = (x1 / orig_w) * 100
            y1_scaled = (y1 / orig_h) * 100
            x2_scaled = (x2 / orig_w) * 100
            y2_scaled = (y2 / orig_h) * 100

            shape = {
                "label": cls,
                "points": [
                    [round(x1_scaled, 2), round(y1_scaled, 2)],
                    [round(x2_scaled, 2), round(y2_scaled, 2)]
                ],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
                "unit": "%"
            }
            shapes.append(shape)

        with open(save_path, "w") as f:
            json.dump(shapes, f, indent=2)

    elif data and isinstance(data[0], dict) and 'bounding_box' in data[0]:
        # New enhanced grouped format
        output_data = {
            "image_dimensions": {"width": orig_w, "height": orig_h},
            "groups": []
        }

        for group in data:
            # Scale bounding box to percentages
            x1, y1, x2, y2 = group['bounding_box']
            x1_scaled = (x1 / orig_w) * 100
            y1_scaled = (y1 / orig_h) * 100
            x2_scaled = (x2 / orig_w) * 100
            y2_scaled = (y2 / orig_h) * 100

            group_data = {
                "type": group["type"],
                "alignment_type": group["alignment_type"],
                "bounding_box": {
                    "x1": round(x1_scaled, 2),
                    "y1": round(y1_scaled, 2),
                    "x2": round(x2_scaled, 2),
                    "y2": round(y2_scaled, 2)
                },
                "elements": []
            }

            for element in group['elements']:
                if element.get("type") == "checkbox_with_label":
                    # Special handling for checkbox with label
                    cb_box = element["combined_box"]
                    element_data = {
                        "type": "checkbox_with_label",
                        "label": "checkbox_with_label",
                        "score": element["score"],
                        "layout": element["layout"],
                        "bounding_box": {
                            "x1": round((cb_box[0] / orig_w) * 100, 2),
                            "y1": round((cb_box[1] / orig_h) * 100, 2),
                            "x2": round((cb_box[2] / orig_w) * 100, 2),
                            "y2": round((cb_box[3] / orig_h) * 100, 2)
                        },
                        "checkbox_box": {
                            "x1": round((element["checkbox"]["x1"] / orig_w) * 100, 2),
                            "y1": round((element["checkbox"]["y1"] / orig_h) * 100, 2),
                            "x2": round((element["checkbox"]["x2"] / orig_w) * 100, 2),
                            "y2": round((element["checkbox"]["y2"] / orig_h) * 100, 2)
                        },
                        "text_box": {
                            "x1": round((element["text"]["x1"] / orig_w) * 100, 2),
                            "y1": round((element["text"]["y1"] / orig_h) * 100, 2),
                            "x2": round((element["text"]["x2"] / orig_w) * 100, 2),
                            "y2": round((element["text"]["y2"] / orig_h) * 100, 2)
                        }
                    }
                else:
                    # Regular element
                    elem_x1, elem_y1, elem_x2, elem_y2 = element["box"]
                    element_data = {
                        "type": "standalone",
                        "label": element["label"],
                        "score": element["score"],
                        "bounding_box": {
                            "x1": round((elem_x1 / orig_w) * 100, 2),
                            "y1": round((elem_y1 / orig_h) * 100, 2),
                            "x2": round((elem_x2 / orig_w) * 100, 2),
                            "y2": round((elem_y2 / orig_h) * 100, 2)
                        }
                    }

                group_data["elements"].append(element_data)

            output_data["groups"].append(group_data)

        with open(save_path, "w") as f:
            json.dump(output_data, f, indent=2)

    else:
        # Old checkbox-text grouped format (from original group_checkbox_with_text function)
        shapes = []
        for element in data:
            if element["type"] == "checkbox_with_label":
                # Save as a special checkbox-with-label element
                cb_box = element["combined_box"]
                x1, y1, x2, y2 = cb_box

                # Scale coords to percentages
                x1_scaled = (x1 / orig_w) * 100
                y1_scaled = (y1 / orig_h) * 100
                x2_scaled = (x2 / orig_w) * 100
                y2_scaled = (y2 / orig_h) * 100

                shape = {
                    "label": "checkbox_with_label",
                    "points": [
                        [round(x1_scaled, 2), round(y1_scaled, 2)],
                        [round(x2_scaled, 2), round(y2_scaled, 2)]
                    ],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {},
                    "unit": "%",
                    "layout": element["layout"],
                    "checkbox_box": {
                        "x1": round((element["checkbox"]["x1"] / orig_w) * 100, 2),
                        "y1": round((element["checkbox"]["y1"] / orig_h) * 100, 2),
                        "x2": round((element["checkbox"]["x2"] / orig_w) * 100, 2),
                        "y2": round((element["checkbox"]["y2"] / orig_h) * 100, 2)
                    },
                    "text_box": {
                        "x1": round((element["text"]["x1"] / orig_w) * 100, 2),
                        "y1": round((element["text"]["y1"] / orig_h) * 100, 2),
                        "x2": round((element["text"]["x2"] / orig_w) * 100, 2),
                        "y2": round((element["text"]["y2"] / orig_h) * 100, 2)
                    }
                }
            else:
                # Regular standalone element
                x1, y1, x2, y2 = element["box"]

                # Scale coords to percentages
                x1_scaled = (x1 / orig_w) * 100
                y1_scaled = (y1 / orig_h) * 100
                x2_scaled = (x2 / orig_w) * 100
                y2_scaled = (y2 / orig_h) * 100

                shape = {
                    "label": element["label"],
                    "points": [
                        [round(x1_scaled, 2), round(y1_scaled, 2)],
                        [round(x2_scaled, 2), round(y2_scaled, 2)]
                    ],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {},
                    "unit": "%"
                }

            shapes.append(shape)

        with open(save_path, "w") as f:
            json.dump(shapes, f, indent=2)


# Debug Function
def debug_checkbox_text_association(final_boxes, image_name, orig_w, orig_h):
    print(f"\n=== Debug: Checkbox-Text Association for {image_name} ===")

    checkboxes = [(i, box) for i, box in enumerate(final_boxes) if box[0] == "checkbox"]
    texts = [(i, box) for i, box in enumerate(final_boxes) if box[0] == "text"]

    print(f"Found {len(checkboxes)} checkboxes and {len(texts)} text elements")

    for cb_idx, checkbox_box in checkboxes:
        cb_cls, cb_score, cb_x1, cb_y1, cb_x2, cb_y2 = checkbox_box
        cb_center_x = (cb_x1 + cb_x2) / 2
        cb_center_y = (cb_y1 + cb_y2) / 2

        print(
            f"\nCheckbox {cb_idx}: center=({cb_center_x:.1f}, {cb_center_y:.1f}), box=({cb_x1:.1f}, {cb_y1:.1f}, {cb_x2:.1f}, {cb_y2:.1f})")

        distances = []
        for text_idx, text_box in texts:
            text_cls, text_score, text_x1, text_y1, text_x2, text_y2 = text_box
            text_center_x = (text_x1 + text_x2) / 2
            text_center_y = (text_y1 + text_y2) / 2

            # Calculate distance
            distance = math.sqrt((cb_center_x - text_center_x) ** 2 + (cb_center_y - text_center_y) ** 2)

            # Check with current parameters
            is_associated = is_text_next_to_checkbox(checkbox_box, text_box)

            distances.append((text_idx, distance, is_associated, text_box))

        # Sort by distance
        distances.sort(key=lambda x: x[1])

        print("  Nearby text elements (sorted by distance):")
        for text_idx, distance, is_associated, text_box in distances[:3]:  # Show top 3 closest
            text_cls, text_score, text_x1, text_y1, text_x2, text_y2 = text_box
            text_center_x = (text_x1 + text_x2) / 2
            text_center_y = (text_y1 + text_y2) / 2

            status = "Associated" if is_associated else "Not associated"
            print(
                f"    Text {text_idx}: distance={distance:.1f}px, center=({text_center_x:.1f}, {text_center_y:.1f}) - {status}")

            # Calculate specific alignment metrics
            horizontal_distance = abs(text_x1 - cb_x2)  # Gap between checkbox right edge and text left edge
            vertical_alignment = abs(cb_center_y - text_center_y)  # Vertical alignment difference

            print(
                f"      Horizontal gap: {horizontal_distance:.1f}px, Vertical alignment diff: {vertical_alignment:.1f}px")


def debug_element_grouping(final_boxes, image_name):
    """Debug function to show how elements are being grouped."""
    print(f"\n=== Debug: Element Grouping for {image_name} ===")

    grouped_elements = group_elements_for_layout(final_boxes)

    print(f"Found {len(grouped_elements)} groups from {len(final_boxes)} original elements")

    for i, group in enumerate(grouped_elements):
        print(f"\nGroup {i}: Type={group['type']}, Alignment={group['alignment_type']}")
        print(f"  Bounding box: {[round(x, 1) for x in group['bounding_box']]}")

        for j, element in enumerate(group['elements']):
            if element.get("type") == "checkbox_with_label":
                print(f"    Element {j}: Checkbox+Label (layout: {element['layout']})")
                print(f"      Combined box: {[round(x, 1) for x in element['combined_box']]}")
            else:
                print(f"    Element {j}: {element['label']} (score: {element['score']:.2f})")
                print(f"      Box: {[round(x, 1) for x in element['box']]}")


# Visualize predictions with enhanced grouping
def visualize_prediction(image_name: str, save_json: bool = True, debug_associations: bool = False,
                         debug_grouping: bool = False, use_enhanced_grouping: bool = True):
    image_path = os.path.join(image_name)
    pil_img, final_boxes = get_predictions(image_path)

    # Debug checkbox-text associations if requested
    if debug_associations:
        debug_checkbox_text_association(final_boxes, image_name, pil_img.width, pil_img.height)

    # Choose grouping method
    if use_enhanced_grouping:
        grouped_elements = group_elements_for_layout(final_boxes)
        if debug_grouping:
            debug_element_grouping(final_boxes, image_name)
    else:
        # Use original grouping method
        checkbox_text_groups = group_checkbox_with_text(final_boxes)
        grouped_elements = [{
            "type": "original_group",
            "elements": checkbox_text_groups,
            "alignment_type": "original",
            "bounding_box": [0, 0, pil_img.width, pil_img.height]
        }]

    # Draw
    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.imshow(pil_img)

    # Color scheme for different group types
    group_colors = ['purple', 'orange', 'cyan', 'magenta', 'yellow', 'lime']

    for group_idx, group in enumerate(grouped_elements):
        group_color = group_colors[group_idx % len(group_colors)]

        # Draw group bounding box
        if len(group['elements']) > 1:
            gx1, gy1, gx2, gy2 = group['bounding_box']
            group_rect = patches.Rectangle(
                (gx1, gy1), gx2 - gx1, gy2 - gy1,
                linewidth=3, edgecolor=group_color, facecolor="none", linestyle=":"
            )
            ax.add_patch(group_rect)

            # Group label
            ax.text(
                gx1, gy1 - 15, f"Group {group_idx} ({group['alignment_type']})",
                color=group_color, fontsize=12, weight="bold",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2)
            )

        # Draw individual elements within the group
        for element in group['elements']:
            if element.get("type") == "checkbox_with_label":
                # Draw combined bounding box in blue
                x1, y1, x2, y2 = element["combined_box"]
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor="blue", facecolor="none", linestyle="--"
                )
                ax.add_patch(rect)

                # Draw individual checkbox in red
                cb = element["checkbox"]
                cb_rect = patches.Rectangle(
                    (cb["x1"], cb["y1"]), cb["x2"] - cb["x1"], cb["y2"] - cb["y1"],
                    linewidth=1, edgecolor="red", facecolor="none"
                )
                ax.add_patch(cb_rect)

                # Draw individual text in green
                txt = element["text"]
                txt_rect = patches.Rectangle(
                    (txt["x1"], txt["y1"]), txt["x2"] - txt["x1"], txt["y2"] - txt["y1"],
                    linewidth=1, edgecolor="green", facecolor="none"
                )
                ax.add_patch(txt_rect)

                # Label
                ax.text(
                    x1, y1 - 5, f"checkbox+label ({element['layout']}) {element['score']:.2f}",
                    color="blue", fontsize=9, weight="bold",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1)
                )
            else:
                # Regular element
                x1, y1, x2, y2 = element["box"]
                color = "darkred" if element["label"] in ["checkbox", "text"] else "red"
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color, facecolor="none"
                )
                ax.add_patch(rect)
                ax.text(
                    x1, y1 - 5, f"{element['label']} {element['score']:.2f}",
                    color=color, fontsize=9, weight="bold",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1)
                )

    ax.axis("off")
    plt.title(f"Enhanced Predictions for {image_name}")
    plt.tight_layout()
    plt.show()

    # Save predictions as JSON
    if save_json:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        json_filename = os.path.splitext(os.path.basename(image_name))[0] + "_enhanced_pred.json"
        save_path = os.path.join(OUTPUT_DIR, json_filename)
        predictions_to_json(grouped_elements, save_path, pil_img.width, pil_img.height)
        print(f"Saved enhanced predictions JSON to: {save_path}")

    return grouped_elements


# Helper function to adjust grouping sensitivity
def adjust_grouping_parameters(horizontal_threshold=None, vertical_threshold=None, alignment_tolerance=None):
    """Adjust the global parameters for element grouping."""
    global HORIZONTAL_GROUP_THRESHOLD, VERTICAL_GROUP_THRESHOLD, ALIGNMENT_TOLERANCE

    if horizontal_threshold is not None:
        HORIZONTAL_GROUP_THRESHOLD = horizontal_threshold
        print(f"Horizontal grouping threshold set to: {HORIZONTAL_GROUP_THRESHOLD}px")

    if vertical_threshold is not None:
        VERTICAL_GROUP_THRESHOLD = vertical_threshold
        print(f"Vertical grouping threshold set to: {VERTICAL_GROUP_THRESHOLD}px")

    if alignment_tolerance is not None:
        ALIGNMENT_TOLERANCE = alignment_tolerance
        print(f"Alignment tolerance set to: {ALIGNMENT_TOLERANCE}px")


# Function to extract layout structure for HTML generation
def extract_layout_structure(grouped_elements, image_width, image_height):
    """Extract a hierarchical layout structure suitable for HTML generation."""
    layout_structure = {
        "container": {
            "width": image_width,
            "height": image_height,
            "rows": []
        }
    }

    # Sort groups by vertical position (top to bottom)
    sorted_groups = sorted(grouped_elements, key=lambda g: g['bounding_box'][1])

    current_row_y = 0
    current_row = []
    row_tolerance = VERTICAL_GROUP_THRESHOLD

    for group in sorted_groups:
        group_y = group['bounding_box'][1]

        # Check if this group should start a new row
        if current_row and abs(group_y - current_row_y) > row_tolerance:
            # Finalize current row
            if current_row:
                # Sort row elements by horizontal position
                current_row.sort(key=lambda g: g['bounding_box'][0])
                layout_structure["container"]["rows"].append({
                    "y_position": current_row_y,
                    "elements": current_row
                })
            current_row = [group]
            current_row_y = group_y
        else:
            # Add to current row
            current_row.append(group)
            if not current_row_y or group_y < current_row_y:
                current_row_y = group_y

    # Don't forget the last row
    if current_row:
        current_row.sort(key=lambda g: g['bounding_box'][0])
        layout_structure["container"]["rows"].append({
            "y_position": current_row_y,
            "elements": current_row
        })

    return layout_structure


# Example usage function
def process_wireframe_with_enhanced_grouping(image_name,
                                             horizontal_threshold=50,
                                             vertical_threshold=30,
                                             alignment_tolerance=40,
                                             debug_mode=False):
    """Process a wireframe image with customizable grouping parameters."""

    # Adjust grouping parameters
    adjust_grouping_parameters(horizontal_threshold, vertical_threshold, alignment_tolerance)

    # Process the image
    grouped_elements = visualize_prediction(
        image_name,
        save_json=True,
        debug_associations=debug_mode,
        debug_grouping=debug_mode,
        use_enhanced_grouping=True
    )

    # Extract layout structure for HTML generation
    pil_img = Image.open(image_name).convert("RGB")
    layout_structure = extract_layout_structure(grouped_elements, pil_img.width, pil_img.height)

    # Save layout structure
    layout_filename = os.path.splitext(image_name)[0] + "_layout_structure.json"
    layout_path = os.path.join(OUTPUT_DIR, layout_filename)
    with open(layout_path, "w") as f:
        json.dump(layout_structure, f, indent=2)
    print(f"Saved layout structure to: {layout_path}")

    return grouped_elements, layout_structure