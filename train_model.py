import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 224
BATCH_SIZE = 16

# Dataset preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    'dataset',
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    'dataset',
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# Load VGG16 model
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze VGG16 layers
for layer in base_model.layers:
    layer.trainable = False

# Create final model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')
])

# Compile model
model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# Save trained model
model.save('model/solar_model.h5')

print('Training Completed!')