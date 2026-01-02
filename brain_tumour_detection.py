
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Set up image generator with rescaling and train/val split
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Set up training data
train = datagen.flow_from_directory('brain_tumor_dataset',
                                    target_size=(128, 128),
                                    batch_size=16,
                                    class_mode='binary',
                                    subset='training')  # 80%

# Set up validation data
val = datagen.flow_from_directory('brain_tumor_dataset',
                                  target_size=(128, 128),
                                  batch_size=16,
                                  class_mode='binary',
                                  subset='validation')  # 20%

# Build CNN model with explicit Input layer to avoid warning
model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary output
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train, validation_data=val, epochs=5)

# Plot training vs validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

# Prediction on a test image
img_path = "test.jpg"
if not os.path.exists(img_path):
    print(f"Error: {img_path} does not exist!")
else:
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    print(f"Prediction score: {prediction[0][0]}")

    # Display the image and result
    plt.imshow(img)
    plt.axis('off')
    if prediction[0][0] > 0.5:
        plt.title("Tumor detected ✅")
        print("Tumor detected ✅")
    else:
        plt.title("No tumor ❌")
        print("No tumor ❌")
    plt.show()
