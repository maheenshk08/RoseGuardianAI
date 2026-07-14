import os 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path = "dataset/Rose/train"
validation_path = "dataset/Rose/validation"
test_path = "dataset/Rose/test"

print("Checking dataset folders...\n")

print("Train:", os.path.exists(train_path))
print("Validation:", os.path.exists(validation_path))
print("Test:", os.path.exists(test_path))

# Create image generators
train_datagen = ImageDataGenerator(rescale=1./255)

validation_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load training dataset
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# Load validation dataset
validation_generator = validation_datagen.flow_from_directory(
    validation_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# Load test dataset
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

print("\nDataset loaded successfully!")

print("\nClasses found:")
print(train_generator.class_indices)

print("\nNumber of training images:", train_generator.samples)
print("Number of validation images:", validation_generator.samples)
print("Number of test images:", test_generator.samples)