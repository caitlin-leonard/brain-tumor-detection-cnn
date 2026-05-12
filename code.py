import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense, Dropout, BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam

# ── Config ──────────────────────────────────────────────────────────────────
DATASET_DIR = 'brain_tumor_dataset'
IMG_SIZE    = (224, 224)   # MobileNetV2 expects 224x224
BATCH_SIZE  = 16
EPOCHS_FROZEN   = 10      # Phase 1: train only top layers
EPOCHS_FINETUNE = 15      # Phase 2: fine-tune last few base layers

# ── Augmentation ─────────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],
    shear_range=0.1,
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
)

train = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True,
    seed=42,
)

val = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False,
    seed=42,
)

print(f"\nClass indices: {train.class_indices}")
print(f"Training samples: {train.samples} | Validation samples: {val.samples}")

# ── Class weights ─────────────────────────────────────────────────────────────
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train.classes),
    y=train.classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

# ── Build model with MobileNetV2 base ────────────────────────────────────────
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,       # remove the ImageNet classification head
    weights='imagenet'       # pretrained weights
)
base_model.trainable = False  # freeze base for Phase 1

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

print(f"\nTotal params: {model.count_params():,}")
print(f"Trainable params (Phase 1): {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# ── Phase 1: Train top layers only ───────────────────────────────────────────
print("\n── Phase 1: Training top layers (base frozen) ──")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
]

history1 = model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS_FROZEN,
    class_weight=class_weight_dict,
    callbacks=callbacks,
)

# ── Phase 2: Fine-tune last 30 layers of base ────────────────────────────────
print("\n── Phase 2: Fine-tuning last 30 layers ──")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),   # much lower LR for fine-tuning
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

history2 = model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS_FINETUNE,
    class_weight=class_weight_dict,
    callbacks=callbacks,
)

# ── Combine histories ─────────────────────────────────────────────────────────
def combine_histories(h1, h2):
    combined = {}
    for key in h1.history:
        combined[key] = h1.history[key] + h2.history[key]
    return combined

history = combine_histories(history1, history2)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(history['accuracy'],     label='Train')
axes[0].plot(history['val_accuracy'], label='Val')
axes[0].axvline(x=len(history1.history['accuracy']), color='gray', linestyle='--', label='Fine-tune start')
axes[0].set_title('Accuracy'); axes[0].legend()

axes[1].plot(history['loss'],     label='Train')
axes[1].plot(history['val_loss'], label='Val')
axes[1].axvline(x=len(history1.history['loss']), color='gray', linestyle='--', label='Fine-tune start')
axes[1].set_title('Loss'); axes[1].legend()

axes[2].plot(history['auc'],     label='Train AUC')
axes[2].plot(history['val_auc'], label='Val AUC')
axes[2].set_title('AUC'); axes[2].legend()

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()

# ── Save model ────────────────────────────────────────────────────────────────
model.save('brain_tumor_mobilenetv2.keras')
print("\nModel saved to brain_tumor_mobilenetv2.keras")

# ── Evaluation ────────────────────────────────────────────────────────────────
val.reset()
preds = (model.predict(val) > 0.5).astype(int).flatten()
true  = val.classes

print("\nClassification Report:")
print(classification_report(true, preds, target_names=list(val.class_indices.keys())))

cm = confusion_matrix(true, preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=val.class_indices.keys(),
            yticklabels=val.class_indices.keys())
plt.title('Confusion Matrix')
plt.ylabel('True'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# ── Single image prediction ───────────────────────────────────────────────────
def predict_image(img_path, model, threshold=0.5):
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found!")
        return
    img     = image.load_img(img_path, target_size=IMG_SIZE)
    arr     = image.img_to_array(img) / 255.0
    arr     = np.expand_dims(arr, axis=0)
    score   = model.predict(arr)[0][0]
    label   = "Tumour Detected" if score > threshold else "No Tumour"
    print(f"Score: {score:.4f}  →  {label}")
    plt.imshow(img); plt.axis('off')
    plt.title(f"{label} (score: {score:.3f})")
    plt.show()

predict_image("test.jpg", model)