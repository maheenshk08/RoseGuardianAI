# Day 3 - Dataset Preparation

## Objective

Today's objective was to prepare the dataset and verify that TensorFlow could successfully read all the rose leaf images required for training the AI model.

## Work Completed

- Selected a suitable Rose Leaf Disease dataset from Kaggle.
- Downloaded and extracted the dataset.
- Added the dataset to the project inside the `dataset` folder.
- Verified the dataset structure (train, validation, and test folders).
- Learned how TensorFlow automatically detects class labels using folder names.
- Created the first AI program in `train_model.py`.
- Imported the required libraries (`os` and `ImageDataGenerator`).
- Configured the paths for the training, validation, and testing datasets.
- Created image generators with image normalization (`rescale=1./255`).
- Loaded the dataset successfully using `flow_from_directory()`.
- Verified that TensorFlow detected all dataset classes correctly.
- Printed the number of training, validation, and testing images.

## Result

The dataset was successfully loaded and is ready for CNN model training.

## Problems Faced

No major issues occurred. TensorFlow displayed informational startup messages, but the dataset loaded successfully.

## Next Goal

Build the Convolutional Neural Network (CNN) architecture and begin training the Rose Guardian AI model.