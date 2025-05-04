import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, Flatten,
                                     Conv2D, AveragePooling2D,
                                     BatchNormalization, LeakyReLU)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Enable GPU memory growth if a GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Parameters
batch_size = 100
epochs = 100
image_size = (100, 100)
input_shape = (100, 100, 1)

# Image data generators
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    'data/train',
    target_size=image_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

test_generator = datagen.flow_from_directory(
    'data/test',
    target_size=image_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

# Build the CNN model
model = Sequential([
    Input(shape=input_shape),
    Conv2D(32, kernel_size=3, activation='relu'),
    AveragePooling2D(pool_size=2),
    Conv2D(64, kernel_size=3, activation='relu'),
    AveragePooling2D(pool_size=2),
    Conv2D(64, kernel_size=3, activation='relu'),
    AveragePooling2D(pool_size=2),
    Conv2D(128, kernel_size=3, activation='relu'),
    AveragePooling2D(pool_size=2),
    BatchNormalization(),
    Dropout(0.25),
    Flatten(),
    Dense(64),
    LeakyReLU(),
    Dropout(0.3),
    Dense(32),
    LeakyReLU(),
    Dense(7, activation='softmax')  # Adjust output units if needed
])

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator
)

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Save the model
model.save('model.h5')
print('Model saved as model.h5')

# Evaluate the model
score = model.evaluate(test_generator)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])
