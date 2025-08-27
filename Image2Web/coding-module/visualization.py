import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Disable GUI backend for Cloud Run
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageOps
import json
import os
import math

model = None # Makes model global variable

# CONFIG
MODEL_PATH = "./best_model.keras"
IMAGE_DIR = "./image/"
OUTPUT_DIR = "./output/"
CLASS_NAMES = ["button", "checkbox", "image", "navbar", "paragraph", "text", "textfield"]

IMG_SIZE = 416
GRID_SIZE = 16
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.1

# Checkbox-Text Association Logic
def is_text_next_to_checkbox(checkbox_box, text_box, max_distance=300, alignment_tolerance=200):
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


def group_checkbox_with_text(final_boxes): # Groups checkboxes with nearby text
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
def predictions_to_json(final_boxes, save_path, orig_w, orig_h):
    # First group checkbox-text pairs
    grouped_elements = group_checkbox_with_text(final_boxes)

    shapes = []
    for element in grouped_elements:
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


# Visualize predictions
def visualize_prediction(image_name: str, save_json: bool = True, debug_associations: bool = False):
    image_path = os.path.join(IMAGE_DIR, image_name)
    pil_img, final_boxes = get_predictions(image_path)

    # Debug checkbox-text associations if requested
    if debug_associations:
        debug_checkbox_text_association(final_boxes, image_name, pil_img.width, pil_img.height)

    # Group elements
    grouped_elements = group_checkbox_with_text(final_boxes)

    # Draw
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(pil_img)

    for element in grouped_elements:
        if element["type"] == "checkbox_with_label":
            # Draw combined bounding box in blue
            x1, y1, x2, y2 = element["combined_box"]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=3, edgecolor="blue", facecolor="none", linestyle="--"
            )
            ax.add_patch(rect)

            # Draw individual checkbox in red
            cb = element["checkbox"]
            cb_rect = patches.Rectangle(
                (cb["x1"], cb["y1"]), cb["x2"] - cb["x1"], cb["y2"] - cb["y1"],
                linewidth=2, edgecolor="red", facecolor="none"
            )
            ax.add_patch(cb_rect)

            # Draw individual text in green
            txt = element["text"]
            txt_rect = patches.Rectangle(
                (txt["x1"], txt["y1"]), txt["x2"] - txt["x1"], txt["y2"] - txt["y1"],
                linewidth=2, edgecolor="green", facecolor="none"
            )
            ax.add_patch(txt_rect)

            # Label
            ax.text(
                x1, y1 - 5, f"checkbox+label ({element['layout']}) {element['score']:.2f}",
                color="blue", fontsize=10, weight="bold",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1)
            )
        else:
            # Regular element
            x1, y1, x2, y2 = element["box"]
            color = "orange" if element["label"] in ["checkbox", "text"] else "red"
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                x1, y1 - 5, f"{element['label']} {element['score']:.2f}",
                color=color, fontsize=10, weight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1)
            )

    ax.axis("off")
    plt.title(f"Predictions for {image_name}")
    plt.show()

    # Save predictions as JSON
    if save_json:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        json_filename = os.path.splitext(image_name)[0] + "_pred.json"
        save_path = os.path.join(OUTPUT_DIR, json_filename)
        predictions_to_json(final_boxes, save_path, pil_img.width, pil_img.height)
        print(f"Saved enhanced predictions JSON to: {save_path}")



# Main
if __name__ == "__main__":
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    visualize_prediction(".jpg", save_json=False, debug_associations=False)