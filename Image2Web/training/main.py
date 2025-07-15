import os
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses
from sklearn.model_selection import train_test_split

# Directory Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_IMAGES = os.path.join(DATASET_DIR, "images", "train")
TRAIN_ANNOTATIONS = os.path.join(DATASET_DIR, "annotations", "train")
VAL_IMAGES = os.path.join(DATASET_DIR, "images", "validate")
VAL_ANNOTATIONS = os.path.join(DATASET_DIR, "annotations", "validate")
TEST_IMAGES = os.path.join(DATASET_DIR, "images", "test")
TEST_ANNOTATIONS = os.path.join(DATASET_DIR, "annotations", "test")

# Preprocessing Functions
def preprocess_image(image_path, img_size=(300, 300)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    processed = cv2.resize(processed, img_size)
    return np.expand_dims(processed, axis=-1).astype(np.float32) / 255.0  # Add channel dim

# Converts readable data from JSON to usable data for TensorFlow/Keras
def parse_labelme_to_tfrecord(json_path):
    with open(json_path) as f:
        data = json.load(f)

    # Get corresponding image path
    img_path = os.path.join(os.path.dirname(json_path), data["imagePath"])
    image = preprocess_image(img_path)

    # Parse annotations
    bboxes = []
    labels = []
    label_to_id = {"button": 0, "h1": 1, "text": 2, "image": 3, "div": 4}  # Update with your classes

    for shape in data["shapes"]:
        if shape["shape_type"] != "rectangle":
            continue

        (x1, y1), (x2, y2) = shape["points"]
        x_min, x_max = min(x1, x2) / data["imageWidth"], max(x1, x2) / data["imageWidth"]
        y_min, y_max = min(y1, y2) / data["imageHeight"], max(y1, y2) / data["imageHeight"]

        bboxes.append([x_min, y_min, x_max, y_max])
        labels.append(label_to_id[shape["label"]])

    return image, np.array(bboxes), np.array(labels)

# Creates TensorFlow Dataset
def create_tf_dataset(annotations_dir, batch_size=8):
    json_files = [os.path.join(annotations_dir, f) for f in os.listdir(annotations_dir) if f.endswith(".json")]

    def generator():
        for json_file in json_files:
            image, bboxes, labels = parse_labelme_to_tfrecord(json_file)
            yield image, {"bboxes": bboxes, "labels": labels}

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(300, 300, 1), dtype=tf.float32),
            {
                "bboxes": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                "labels": tf.TensorSpec(shape=(None,), dtype=tf.int32)
            }
        )
    )
    return dataset.padded_batch(
        batch_size,
        padded_shapes=(
            (300, 300, 1),
            {"bboxes": (None, 4), "labels": (None,)}
        )
    ).prefetch(tf.data.AUTOTUNE)


# Custom Model
def build_custom_ssd(input_shape=(300, 300, 1), num_classes=5):
    inputs = layers.Input(shape=input_shape)

    # Backbone
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Detection heads
    head1 = layers.Conv2D(6 * (4 + num_classes), (3, 3), padding='same')(x)
    head1 = layers.Reshape((-1, 4 + num_classes))(head1)

    x_small = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    head2 = layers.Conv2D(6 * (4 + num_classes), (3, 3), padding='same')(x_small)
    head2 = layers.Reshape((-1, 4 + num_classes))(head2)

    outputs = layers.Concatenate(axis=1)([head1, head2])
    return Model(inputs=inputs, outputs=outputs)


# Loss
class SSDLoss(losses.Loss):
    def call(self, y_true, y_pred):
        # Classification loss (sparse categorical crossentropy)
        cls_loss = losses.SparseCategoricalCrossentropy(from_logits=True)(
            y_true["labels"], y_pred[..., :5]
        )
        # Localization loss (smooth L1)
        box_loss = losses.Huber()(y_true["bboxes"], y_pred[..., 5:])
        return cls_loss + box_loss

def main():
    # Initialize model
    model = build_custom_ssd(num_classes=5)
    model.compile(optimizer=optimizers.Adam(1e-4), loss=SSDLoss())

    # Load datasets
    train_dataset = create_tf_dataset(TRAIN_ANNOTATIONS)
    val_dataset = create_tf_dataset(VAL_ANNOTATIONS)

    # Train
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint("model.h5", save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=5)
        ]
    )

    # Test (optional)
    test_dataset = create_tf_dataset(TEST_ANNOTATIONS)
    test_loss = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}")

    # Save final model
    model.save("saved_model")


if __name__ == "__main__":
    main()