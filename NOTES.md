# 🌹 Rose Guardian AI - Learning Notes

---

# Project Information

## Project Name
Rose Guardian AI

## Project Goal
Develop an AI-powered web application that detects diseases in rose plants from leaf images, estimates disease severity, analyzes weather conditions, and provides recommendations to maintain healthy rose plants.

---

# Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Main programming language |
| TensorFlow/Keras | Build and train AI model |
| OpenCV | Image processing |
| Flask | Backend web framework |
| HTML | Website structure |
| CSS | Website styling |
| JavaScript | User interaction |
| NumPy | Numerical computations |
| Weather API | Fetch temperature and humidity |
| Git | Version control |
| GitHub | Project hosting |

---

# Folder Purpose

## dataset/
Stores all the rose leaf images used for training, validation, and testing.

## models/
Stores the trained AI model after training.

## uploads/
Stores images uploaded by the user for prediction.

## static/
Stores CSS, JavaScript, images, and other static files.

## templates/
Stores HTML pages used by Flask.

## Project Diary/
Contains daily progress reports.

## README.md
Project documentation displayed on GitHub.

## requirements.txt
Contains the list of Python libraries required to run the project.

---

# Python Environment

## Why Python 3.11?

Python 3.11 is widely supported by AI libraries like TensorFlow and provides better compatibility than newer Python versions.

## What is Anaconda?

Anaconda is a Python distribution mainly used for Artificial Intelligence, Machine Learning, Data Science, and scientific computing.

## What is pip?

pip is Python's package manager.

Example:

python -m pip install flask

---

# Developer Concepts

## What is Git?

Git is a Version Control System that tracks changes made in a project.

## What is GitHub?

GitHub is an online platform where Git projects are stored and shared.

Difference:

Git → Local version control

GitHub → Online repository

---

# Project Architecture

User

↓

Upload Rose Leaf Image

↓

Flask Website

↓

AI Model

↓

Disease Prediction

↓

Weather Analysis

↓

Recommendation Engine

↓

Result Page

# Virtual Environment

## What is a Virtual Environment?

A virtual environment is an isolated Python environment created for a specific project. It allows each project to have its own Python libraries without affecting other projects.

## Advantages

- Prevents library conflicts.
- Keeps projects organized.
- Makes the project easier to share.
- Allows different projects to use different library versions.

# Day 3 Notes

## What I Learned

### Dataset Structure

Machine learning datasets are commonly divided into:

- Train
- Validation
- Test

### ImageDataGenerator

`ImageDataGenerator` automatically:

- Reads images from folders.
- Assigns labels based on folder names.
- Resizes images.
- Normalizes pixel values.
- Feeds images to the CNN during training.

### Image Normalization

Images contain pixel values between 0 and 255.

Using:

```python
rescale = 1./255
```

converts pixel values to the range 0–1, making training more stable.

### Dataset Paths

The project uses three dataset paths:

- dataset/Rose/train
- dataset/Rose/validation
- dataset/Rose/test

### Class Labels

TensorFlow detected:

- Healthy_Leaf_Rose
- Rose_Rust
- Rose_sawfly_Rose_slug

without manually assigning labels because it uses the folder names.

## Data Pipeline

### What is a Pipeline?

A pipeline is a sequence of connected steps where the output of one step becomes the input of the next step. In machine learning, a pipeline helps automate the process of preparing data before training the AI model.

### Dataset Pipeline in RoseGuardian AI

The dataset pipeline used in this project is:

Dataset
↓
Python locates the dataset folders
↓
TensorFlow reads the images
↓
Images are resized to 224 × 224 pixels
↓
Pixel values are normalized from 0–255 to 0–1
↓
Class labels are assigned automatically from folder names
↓
Images become ready for CNN training

### Components Used

- `os` module – Locates dataset folders.
- `ImageDataGenerator` – Reads and preprocesses images.
- `flow_from_directory()` – Loads images and automatically assigns labels.
- `rescale=1./255` – Normalizes pixel values for better model training.

### Why is the Dataset Pipeline Important?

- Ensures the dataset is organized correctly.
- Reduces the chance of training errors.
- Automatically prepares thousands of images for AI.
- Makes the training process faster and more reliable.

# 📚 Day 4 Notes – Building the AI Model

---

# 1. Transfer Learning

Transfer Learning is a machine learning technique where a pretrained model is reused for a new task instead of training a neural network from scratch.

Instead of learning basic image features again, the model uses the knowledge it already gained from millions of images and learns only the new classes.

### Advantages
- Faster training
- Better accuracy
- Requires fewer images
- Reduces overfitting
- Saves computational resources

---

# 2. MobileNetV2

MobileNetV2 is a lightweight Convolutional Neural Network (CNN) developed by Google.

It is already trained on the ImageNet dataset and is widely used for image classification projects because it is fast and efficient.

In RoseGuardianAI, MobileNetV2 is used as the feature extractor.

```python
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)
```

---

# 3. ImageNet

ImageNet is one of the world's largest image datasets.

It contains:
- Over 1 million images
- Around 1000 different classes

MobileNetV2 has already learned general image features from this dataset.

---

# 4. include_top=False

The original MobileNetV2 model ends with a classifier that predicts 1000 ImageNet classes.

Since RoseGuardianAI only predicts 3 classes, we remove that classifier.

```python
include_top=False
```

This allows us to create our own custom classifier.

---

# 5. Freezing the Base Model

```python
base_model.trainable = False
```

Freezing means the pretrained MobileNetV2 weights will not change during training.

Only the newly added classifier will learn from our rose dataset.

Benefits:
- Faster training
- Prevents destroying pretrained knowledge
- Better performance on smaller datasets

---

# 6. Sequential Model

Sequential is the simplest way to build a neural network by stacking layers one after another.

Example:

Input Image
↓

MobileNetV2
↓

GlobalAveragePooling2D
↓

Dense Layer
↓

Dropout
↓

Output Layer

---

# 7. GlobalAveragePooling2D

Converts large feature maps into a compact feature vector.

Benefits:
- Reduces the number of parameters
- Prevents overfitting
- Makes the model more efficient

---

# 8. Dense Layer

Dense is a fully connected neural network layer.

It learns patterns from the extracted features.

Example:

```python
Dense(256, activation="relu")
```

256 neurons are used for learning.

---

# 9. ReLU Activation Function

ReLU stands for Rectified Linear Unit.

Formula:

f(x) = max(0, x)

Advantages:
- Fast computation
- Solves vanishing gradient problems
- Most commonly used hidden layer activation

---

# 10. Dropout Layer

```python
Dropout(0.5)
```

Dropout randomly disables 50% of neurons during training.

Purpose:
- Prevents overfitting
- Improves generalization
- Makes the model more robust

---

# 11. Softmax Activation

Softmax converts the final outputs into probabilities.

Example:

Healthy Leaf = 97%

Rose Rust = 2%

Sawfly = 1%

The class with the highest probability becomes the prediction.

---

# 12. Model Compilation

Before training, the model must be compiled.

```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

Compilation defines:
- Optimizer
- Loss Function
- Evaluation Metric

---

# 13. Adam Optimizer

Adam is one of the most popular optimization algorithms.

Purpose:
- Updates model weights
- Helps the model learn efficiently
- Provides faster convergence

Learning Rate Used:

0.001

---

# 14. Loss Function

Loss measures how wrong the model's prediction is.

For multi-class classification we use:

```python
categorical_crossentropy
```

Goal:

Lower Loss = Better Model

---

# 15. Accuracy

Accuracy measures the percentage of correct predictions.

Example:

100 Images Tested

95 Correct Predictions

Accuracy = 95%

---

# 16. Model Summary

The Model Summary displays:

- Model Architecture
- Layer Names
- Output Shapes
- Number of Parameters
- Trainable Parameters
- Non-Trainable Parameters

Our Model Summary:

Total Parameters:
2,586,691

Trainable Parameters:
328,707

Non-Trainable Parameters:
2,257,984

---

# 17. RoseGuardianAI Model Architecture

Input Image (224 × 224 × 3)

↓

MobileNetV2 (Feature Extraction)

↓

GlobalAveragePooling2D

↓

Dense (256, ReLU)

↓

Dropout (0.5)

↓

Dense (3, Softmax)

↓

Prediction

Healthy Leaf

Rose Rust

Sawfly Damage

---

# Key Learnings

✅ Learned Transfer Learning

✅ Used MobileNetV2 as a pretrained model

✅ Froze pretrained layers

✅ Built a custom classifier

✅ Compiled the model

✅ Understood Optimizer, Loss and Accuracy

✅ Successfully generated the model summary

✅ Model is now ready for training