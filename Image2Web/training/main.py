from __future__ import annotations
import argparse
import json
import os
import random
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageOps


# Configuration / Hyperparams
DEFAULT_CLASSES = [
    "button",
    "checkbox",
    "image",
    "navbar",
    "paragraph",
    "text",
    "textfield"
]


@dataclass
class Config:
    data_root: str
    workdir: str
    img_size: int = 416
    grid_size: int = 16
    boxes_per_cell: int = 1
    epochs: int = 40
    batch_size: int = 4
    base_lr: float = 2e-4
    weight_decay: float = 1e-4
    jitter: float = 0.1
    flip_prob: float = 0.3
    seed: int = 1337


# Utilities
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_class_list(data_root: str) -> List[str]:
    path = os.path.join(data_root, "classes.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            classes = [ln.strip() for ln in f if ln.strip()]
        return classes
    return DEFAULT_CLASSES


# LabelMe parsing
def _load_labelme_boxes(json_path: str, img_w: int, img_h: int) -> List[Tuple[str, List[float]]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    boxes = []
    for shp in data.get("shapes", []):
        label = shp.get("label", "").strip().lower()  # Normalize to lowercase
        pts = shp.get("points", [])
        if not pts or len(pts) < 2:
            continue

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        x0, y0 = float(min(xs)), float(min(ys))
        x1, y1 = float(max(xs)), float(max(ys))

        # Normalize to [0, 1]
        x0, y0 = x0 / img_w, y0 / img_h
        x1, y1 = x1 / img_w, y1 / img_h

        # Ensure valid bounds
        x0 = max(0.0, min(1.0, x0))
        y0 = max(0.0, min(1.0, y0))
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))

        # Ensure minimum box size
        min_size = 0.02  # 2% of image
        if (x1 - x0) >= min_size and (y1 - y0) >= min_size:
            boxes.append((label, [x0, y0, x1, y1]))

    return boxes


# Grid assignment
def _assign_to_grid(boxes_xyxy: np.ndarray, classes: np.ndarray, S: int, num_classes: int) -> np.ndarray:
    target = np.zeros((S, S, 5 + num_classes), dtype=np.float32)
    cell_size = 1.0 / S

    for (x0, y0, x1, y1), c in zip(boxes_xyxy, classes):
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        w = x1 - x0
        h = y1 - y0

        # Ensure reasonable box size
        w = max(0.02, min(0.8, w))  # Clamp between 2% and 80%
        h = max(0.02, min(0.8, h))

        gi = int(min(S - 1, max(0, cx * S)))
        gj = int(min(S - 1, max(0, cy * S)))

        local_x = (cx - gi * cell_size) / cell_size
        local_y = (cy - gj * cell_size) / cell_size

        # Better clamping
        local_x = max(0.01, min(0.99, local_x))
        local_y = max(0.01, min(0.99, local_y))

        # Always assign
        target[gj, gi, 0] = 1.0
        target[gj, gi, 1] = local_x
        target[gj, gi, 2] = local_y
        target[gj, gi, 3] = w
        target[gj, gi, 4] = h

        # Class (one-hot)
        target[gj, gi, 5:] = 0.0
        target[gj, gi, 5 + int(c)] = 1.0

    return target


# Build dataset
def build_dataset_fixed(cfg: Config, split: str = "train") -> tf.data.Dataset:
    img_dir = os.path.join(cfg.data_root, "wireframes")
    ann_dir = os.path.join(cfg.data_root, "annotations")

    # Find all valid pairs
    img_paths = sorted(glob(os.path.join(img_dir, "*.jpg")) +
                       glob(os.path.join(img_dir, "*.png")) +
                       glob(os.path.join(img_dir, "*.jpeg")))

    valid_pairs = []
    for img_path in img_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(ann_dir, f"{stem}.json")
        if os.path.exists(json_path):
            valid_pairs.append((img_path, json_path))

    print(f"Found {len(valid_pairs)} valid pairs")
    assert valid_pairs, "No valid image-annotation pairs found"

    # Split data
    random.shuffle(valid_pairs)
    n = len(valid_pairs)
    n_train = max(int(0.85 * n), n - 30) if n > 30 else max(1, n - 5)

    if split == "train":
        pairs = valid_pairs[:n_train]
    else:
        pairs = valid_pairs[n_train:]

    print(f"Using {len(pairs)} pairs for {split}")

    # Load classes
    class_list = load_class_list(cfg.data_root)
    class_to_id = {c.lower(): i for i, c in enumerate(class_list)}  # Normalize to lowercase
    num_classes = len(class_list)

    print(f"Classes: {class_list}")

    def _generator():
        for img_path, json_path in pairs:
            try:
                # Load and process image
                pil_img = Image.open(img_path).convert("RGB")
                pil_img = ImageOps.exif_transpose(pil_img)
                orig_w, orig_h = pil_img.size

                # Load boxes
                boxes = _load_labelme_boxes(json_path, orig_w, orig_h)

                # Filter valid classes and collect stats
                labels, bxs = [], []
                for lab, xyxy in boxes:
                    if lab in class_to_id:
                        labels.append(class_to_id[lab])
                        bxs.append(xyxy)

                if not bxs:  # Skip if no valid boxes
                    continue

                # Resize image
                pil_img_resized = pil_img.resize((cfg.img_size, cfg.img_size), Image.LANCZOS)
                img = np.array(pil_img_resized, dtype=np.float32) / 255.0

                # Create target
                bxs_array = np.array(bxs, dtype=np.float32)
                labels_array = np.array(labels, dtype=np.int32)
                target = _assign_to_grid(bxs_array, labels_array, cfg.grid_size, num_classes)

                yield img, target

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    # Create dataset
    ds = tf.data.Dataset.from_generator(
        _generator,
        output_signature=(
            tf.TensorSpec(shape=(cfg.img_size, cfg.img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(cfg.grid_size, cfg.grid_size, 5 + num_classes), dtype=tf.float32),
        ),
    )

    # Apply augmentation
    if split == "train":
        def _augment(img, target):
            # Simple horizontal flip
            if tf.random.uniform([]) < cfg.flip_prob:
                img = tf.image.flip_left_right(img)
                target = tf.reverse(target, axis=[1])

                # Fix x coordinates
                obj_mask = target[..., 0:1] > 0.5
                x_coords = target[..., 1:2]
                flipped_x = 1.0 - x_coords
                target = tf.concat([
                    target[..., 0:1],
                    tf.where(obj_mask, flipped_x, x_coords),
                    target[..., 2:]
                ], axis=-1)

            # Random grayscale
            if tf.random.uniform([]) < 0.3:
                g = tf.image.rgb_to_grayscale(img)
                img = tf.image.grayscale_to_rgb(g)

            # Gaussian noise
            if tf.random.uniform([]) < 0.5:
                noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.02)
                img = img + noise

            # Bleed-through lines simulation
            if tf.random.uniform([]) < 0.3:
                shifted = tf.roll(img, shift=tf.random.uniform([], -5, 6, dtype=tf.int32), axis=1)
                shifted = tf.roll(shifted, shift=tf.random.uniform([], -5, 6, dtype=tf.int32), axis=0)
                ghost = 1.0 - shifted
                img = tf.clip_by_value(img * 0.9 + ghost * 0.1, 0.0, 1.0)

            # Light brightness/contrast
            img = tf.image.random_brightness(img, 0.1)
            img = tf.image.random_contrast(img, 0.9, 1.1)
            img = tf.clip_by_value(img, 0.0, 1.0)

            return img, target

        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=170, seed=cfg.seed)

    ds = ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# Build model
def build_simple_model(cfg: Config, num_classes: int):
    S = cfg.grid_size
    inputs = keras.Input(shape=(cfg.img_size, cfg.img_size, 3))

    x = inputs
    x = conv_block(x, 32, 3, 1)
    x = conv_block(x, 32, 3, 1)
    x = layers.MaxPool2D(2)(x)

    x = conv_block(x, 64, 3, 1)
    x = conv_block(x, 64, 3, 1)
    x = layers.MaxPool2D(2)(x)

    x = conv_block(x, 128, 3, 1)
    x = conv_block(x, 128, 3, 1)
    x = layers.MaxPool2D(2)(x)

    x = conv_block(x, 256, 3, 1)
    x = conv_block(x, 256, 3, 1)
    x = layers.MaxPool2D(2)(x)

    x = conv_block(x, 512, 3, 1)
    x = conv_block(x, 256, 3, 1)

    # Ensure we get to grid size
    current_size = cfg.img_size // 16  # After 4 maxpools
    while current_size > S:
        x = layers.MaxPool2D(2)(x)
        current_size //= 2

    if x.shape[1] != S or x.shape[2] != S:
        x = layers.Resizing(S, S)(x)

    # Detection head - simplified
    x = conv_block(x, 256, 3, 1)
    x = conv_block(x, 128, 3, 1)

    # Final prediction
    outputs = layers.Conv2D(5 + num_classes, 1, padding="same", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="simple_wireframe_detector")
    return model


def conv_block(x, filters, kernel_size, stride, activation='relu'):
    x = layers.Conv2D(filters, kernel_size, stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if activation == 'relu':
        x = layers.ReLU()(x)
    elif activation == 'leaky_relu':
        x = layers.LeakyReLU(0.1)(x)
    return x


# Loss calculation
class LossCalculation(tf.keras.losses.Loss):
    def __init__(self, num_classes, lambda_coord=5.0, lambda_noobj=0.5, name='loss_calculation'):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def call(self, y_true, y_pred):
        # Split predictions and targets
        obj_true = y_true[..., 0]
        box_true = y_true[..., 1:5]
        cls_true = y_true[..., 5:]

        obj_pred_logits = y_pred[..., 0]
        box_pred = y_pred[..., 1:5]
        cls_pred_logits = y_pred[..., 5:]

        # Create masks
        obj_mask = tf.cast(obj_true > 0.5, tf.float32)
        noobj_mask = 1.0 - obj_mask

        num_pos = tf.reduce_sum(obj_mask)
        num_pos = tf.maximum(num_pos, 1.0)  # Avoid division by zero

        # 1. Objectness loss
        obj_loss_pos = obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=obj_true, logits=obj_pred_logits)
        obj_loss_neg = noobj_mask * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=obj_true, logits=obj_pred_logits)

        obj_loss = (tf.reduce_sum(obj_loss_pos) + self.lambda_noobj * tf.reduce_sum(obj_loss_neg)) / tf.cast(tf.size(obj_true), tf.float32)

        # 2. Box regression loss
        xy_pred = tf.nn.sigmoid(box_pred[..., 0:2])
        wh_pred = tf.nn.sigmoid(box_pred[..., 2:4])

        xy_true = box_true[..., 0:2]
        wh_true = box_true[..., 2:4]

        # Use smooth L1 loss for stability
        xy_loss = tf.reduce_sum(obj_mask[..., tf.newaxis] * self._smooth_l1_loss(xy_true - xy_pred)) / num_pos
        wh_loss = tf.reduce_sum(obj_mask[..., tf.newaxis] * self._smooth_l1_loss(wh_true - wh_pred)) / num_pos

        box_loss = self.lambda_coord * (xy_loss + wh_loss)

        # 3. Classification loss
        cls_loss = tf.reduce_sum(obj_mask * tf.nn.softmax_cross_entropy_with_logits(
            labels=cls_true, logits=cls_pred_logits)) / num_pos

        total_loss = obj_loss + box_loss + cls_loss

        # Regularization
        total_loss = tf.clip_by_value(total_loss, 0.0, 100.0)

        return total_loss

    def _smooth_l1_loss(self, x, beta=1.0):
        abs_x = tf.abs(x)
        return tf.where(abs_x < beta, 0.5 * x * x / beta, abs_x - 0.5 * beta)


# Extract predictions
def extract_predictions(pred_grid, class_list, conf_threshold=0.1): # Adjust confidence threshold
    boxes = []
    S = pred_grid.shape[0]
    cell_size = 1.0 / S

    for row in range(S):
        for col in range(S):
            # Get objectness score
            obj_score = float(tf.nn.sigmoid(pred_grid[row, col, 0]))

            # Get class info
            class_logits = pred_grid[row, col, 5:]
            class_probs = tf.nn.softmax(class_logits).numpy()
            class_id = int(np.argmax(class_probs))
            class_conf = float(class_probs[class_id])

            final_score = obj_score * class_conf

            # Use very low threshold for debugging
            if final_score < conf_threshold:
                continue

            # Get box coordinates
            x_offset = float(tf.nn.sigmoid(pred_grid[row, col, 1]))
            y_offset = float(tf.nn.sigmoid(pred_grid[row, col, 2]))
            width = float(tf.nn.sigmoid(pred_grid[row, col, 3]))
            height = float(tf.nn.sigmoid(pred_grid[row, col, 4]))

            # Convert to absolute coordinates
            center_x = (col + x_offset) * cell_size
            center_y = (row + y_offset) * cell_size

            x1 = max(0, min(1, center_x - width / 2))
            y1 = max(0, min(1, center_y - height / 2))
            x2 = max(0, min(1, center_x + width / 2))
            y2 = max(0, min(1, center_y + height / 2))

            if x2 > x1 and y2 > y1:
                class_name = class_list[class_id] if class_id < len(class_list) else f"class_{class_id}"
                boxes.append((class_name, final_score, x1, y1, x2, y2))

    return boxes


# mAP callback
class mAP_callback(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset, num_classes, class_names):
        super().__init__()
        self.val_dataset = val_dataset
        self.num_classes = num_classes
        self.class_names = class_names

    def on_epoch_end(self, epoch, logs=None):
        # Compute mAP every 5 epochs
        if epoch % 5 != 0:
            return

        print(f"\n--- Computing mAP for epoch {epoch + 1} ---")

        all_true = defaultdict(list)
        all_pred = defaultdict(list)

        sample_count = 0
        for images, labels in self.val_dataset.take(2):  # Limit for debugging
            preds = self.model.predict(images, verbose=0)

            for i in range(len(images)):
                if sample_count >= 8:  # Process only first 8 samples
                    break

                # Extract GT
                gt_boxes = self._extract_gt_boxes(labels[i].numpy())
                for cls_id, box in gt_boxes:
                    all_true[cls_id].append(box)

                # Extract predictions with VERY low threshold for debugging
                pred_boxes = extract_predictions(preds[i], self.class_names, conf_threshold=0.01)
                for class_name, score, x1, y1, x2, y2 in pred_boxes:
                    if class_name in self.class_names:
                        cls_id = self.class_names.index(class_name)
                        all_pred[cls_id].append((score, [x1, y1, x2, y2]))

                sample_count += 1

            if sample_count >= 8:
                break

        # Compute and display results
        ap_scores = []
        for c in range(self.num_classes):
            gt_boxes = all_true.get(c, [])
            pred_boxes = all_pred.get(c, [])

            if gt_boxes:
                ap = self._compute_ap(gt_boxes, pred_boxes) if pred_boxes else 0.0
                ap_scores.append(ap)
                print(f"  {self.class_names[c]}: AP@0.5={ap:.4f} (GT:{len(gt_boxes)}, Pred:{len(pred_boxes)})")
            else:
                print(f"  {self.class_names[c]}: No GT boxes")

        mAP = np.mean(ap_scores) if ap_scores else 0.0
        print(f"\nEpoch {epoch + 1} - mAP@0.5: {mAP:.4f}")

        if logs:
            logs["mAP"] = mAP

        # Debug info
        total_preds = sum(len(all_pred.get(c, [])) for c in range(self.num_classes))
        total_gts = sum(len(all_true.get(c, [])) for c in range(self.num_classes))
        print(f"Debug: Total predictions={total_preds}, Total GT={total_gts}")

    def _extract_gt_boxes(self, label_grid):
        boxes = []
        S = label_grid.shape[0]
        cell_size = 1.0 / S

        for row in range(S):
            for col in range(S):
                if label_grid[row, col, 0] > 0.5:
                    x_offset = label_grid[row, col, 1]
                    y_offset = label_grid[row, col, 2]
                    width = label_grid[row, col, 3]
                    height = label_grid[row, col, 4]
                    class_id = np.argmax(label_grid[row, col, 5:])

                    center_x = (col + x_offset) * cell_size
                    center_y = (row + y_offset) * cell_size

                    x1 = max(0, min(1, center_x - width / 2))
                    y1 = max(0, min(1, center_y - height / 2))
                    x2 = max(0, min(1, center_x + width / 2))
                    y2 = max(0, min(1, center_y + height / 2))

                    if x2 > x1 and y2 > y1:
                        boxes.append((class_id, [x1, y1, x2, y2]))

        return boxes

    def _compute_ap(self, true_boxes, pred_boxes_with_scores, iou_threshold=0.5):
        if not pred_boxes_with_scores:
            return 0.0

        pred_boxes_with_scores.sort(key=lambda x: x[0], reverse=True)
        tp = np.zeros(len(pred_boxes_with_scores))
        fp = np.zeros(len(pred_boxes_with_scores))
        matched = set()

        for i, (score, pred_box) in enumerate(pred_boxes_with_scores):
            best_iou = 0
            best_gt = -1

            for j, true_box in enumerate(true_boxes):
                if j in matched:
                    continue
                iou = self._compute_iou(pred_box, true_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = j

            if best_iou >= iou_threshold:
                matched.add(best_gt)
                tp[i] = 1
            else:
                fp[i] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / len(true_boxes)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        # Compute AP using 11-point interpolation
        ap = 0
        for t in np.linspace(0, 1, 11):
            mask = recalls >= t
            if np.any(mask):
                ap += np.max(precisions[mask]) / 11

        return ap

    def _compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / (union_area + 1e-6)


# Training
def train(cfg: Config):
    # Set seed
    set_seed(cfg.seed)

    # Load classes
    class_list = load_class_list(cfg.data_root)
    num_classes = len(class_list)
    print(f"Training with {num_classes} classes: {class_list}")

    # Build model
    model = build_simple_model(cfg, num_classes)
    print(f"Model built with output shape: {model.output_shape}")

    # Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=cfg.base_lr)

    # Loss
    loss_fn = LossCalculation(num_classes=num_classes)

    model.compile(optimizer=optimizer, loss=loss_fn)

    # Build datasets
    print("Building datasets...")
    train_ds = build_dataset_fixed(cfg, split="train")
    val_ds = build_dataset_fixed(cfg, split="val")

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(cfg.workdir, "best_model.keras"),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=25,  # Adjust patience; epochs
            restore_best_weights=True,
            verbose=1,
            min_delta=0.0005
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.8,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),
        mAP_callback(
            val_dataset=val_ds,
            num_classes=num_classes,
            class_names=class_list
        )
    ]

    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history


# CLI
def parse_args() -> Config:
    ap = argparse.ArgumentParser(description="Train Wireframe Detection Model")
    ap.add_argument('--data-root', type=str, required=True,
                    help='Dataset root with wireframes/ and annotations/')
    ap.add_argument('--workdir', type=str, default='./runs/wireframe_fixed',
                    help='Where to write logs/checkpoints')
    ap.add_argument('--img-size', type=int, default=416,
                    help='Input image size')
    ap.add_argument('--grid', type=int, default=16,
                    help='Grid size for detection')
    ap.add_argument('--batch', type=int, default=4,
                    help='Batch size')
    ap.add_argument('--epochs', type=int, default=40,
                    help='Number of training epochs')
    ap.add_argument('--lr', type=float, default=5e-4,
                    help='Learning rate')
    ap.add_argument('--seed', type=int, default=1337,
                    help='Random seed')
    args = ap.parse_args()

    cfg = Config(
        data_root=args.data_root,
        workdir=args.workdir,
        img_size=args.img_size,
        grid_size=args.grid,
        batch_size=args.batch,
        epochs=args.epochs,
        base_lr=args.lr,
        seed=args.seed,
    )
    return cfg


def main():
    print("=== FIXED Wireframe Detection Training ===")

    cfg = parse_args()
    ensure_dir(cfg.workdir)

    print(f"Configuration:")
    print(f"  Data root: {cfg.data_root}")
    print(f"  Work directory: {cfg.workdir}")
    print(f"  Image size: {cfg.img_size}")
    print(f"  Grid size: {cfg.grid_size}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Epochs: {cfg.epochs}")
    print(f"  Learning rate: {cfg.base_lr}")

    # Train model
    model, history = train(cfg)

    print("\n=== Training Summary ===")
    if history.history:
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        print(f"Final training loss: {final_train_loss:.4f}")
        print(f"Final validation loss: {final_val_loss:.4f}")


if __name__ == '__main__':
    main()