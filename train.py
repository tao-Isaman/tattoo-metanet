import tensorflow as tf
from tensorflow.keras import layers, models, applications
import os

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Ensure GPU is enabled
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def load_and_split_data(data_directory, img_height, img_width, batch_size, val_split):
    # Load the dataset
    full_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_directory,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = full_ds.class_names  # Access class names before transformations

    # Create train, validation, and test splits
    val_size = int(len(full_ds) * val_split)
    train_ds = full_ds.skip(val_size)
    val_ds = full_ds.take(val_size)

    # Further split the validation set to create a test set
    test_size = int(len(val_ds) * 0.5)
    test_ds = val_ds.take(test_size)
    val_ds = val_ds.skip(test_size)

    return train_ds, val_ds, test_ds, class_names

def configure_for_performance(ds):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    return ds

# Load and split the data
data_directory = '/content/drive/MyDrive/ภาพรอยสัก'
img_height = 299  # Image size for InceptionResNetV2
img_width = 299
batch_size = 32  # Batch size
val_split = 0.2  # 20% of the data for validation (which includes the test set)
train_ds, val_ds, test_ds, class_names = load_and_split_data(data_directory, img_height, img_width, batch_size, val_split)

# Apply data augmentation to the training dataset
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2)  # Added more augmentations
])

# Configure datasets for performance
train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
test_ds = configure_for_performance(test_ds)

# Define the model architecture with InceptionResNetV2 and data augmentation
inputs = layers.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
base_model = applications.InceptionResNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)
model = models.Model(inputs, outputs)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Model summary
model.summary()

# Implement early stopping and learning rate reduction on plateau
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Train the model with the top layers frozen
history = model.fit(train_ds, epochs=50, validation_data=val_ds, callbacks=[early_stopping, reduce_lr])

# Unfreeze the base model for fine-tuning
base_model.trainable = True

# Recompile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Continue training the model with fine-tuning
history_fine = model.fit(train_ds, epochs=50, validation_data=val_ds, callbacks=[early_stopping, reduce_lr])

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc}")

# Optionally, plot training and validation accuracy
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.plot(history_fine.history['accuracy'], label='fine-tuned accuracy')
plt.plot(history_fine.history['val_accuracy'], label='fine-tuned validation accuracy')
plt.xlabel('Epoch')