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