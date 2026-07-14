# Day 4 – Building the AI Model

Date: 14 July 2026

Today I started building the actual AI model for RoseGuardianAI.

First, I learned the concept of Transfer Learning and understood why pretrained models are preferred over training a deep neural network from scratch.

I selected MobileNetV2 as the base model because it is lightweight, accurate, and already trained on millions of images from the ImageNet dataset.

The original ImageNet classifier was removed using include_top=False so that the model could be customized for the rose leaf disease classification task.

The pretrained MobileNetV2 layers were frozen to preserve the learned features while allowing only the custom classifier to learn from the rose dataset.

A custom classification head was created using GlobalAveragePooling2D, Dense, Dropout, and Softmax layers.

Finally, the model was compiled using the Adam optimizer, categorical crossentropy loss, and accuracy as the evaluation metric.

The model architecture was successfully generated and verified.

Status:
Model architecture completed successfully and ready for training.