# Club-1
Fall Detection YOLO Trainer

This repository contains a YOLOv8-based fall detection training pipeline. It automatically prepares your dataset, trains a YOLO model, and packages the results for deployment.

Features

Auto-install required packages (ultralytics, pandas, PyYAML, opencv-python-headless)
Automatically extracts dataset ZIP and organizes train and validation folders
Matches images with YOLO-style labels (.txt)
Splits dataset into 80% training / 20% validation
Trains YOLOv8 model on CPU or GPU
Tracks accuracy using mAP@0.5
Packages trained model, YAML, and training plots into a ZIP

How it works (step by step)

Checks and installs required packages.
Sets project paths, class names (fall), and chooses device (GPU if available).
Creates folders for YOLO’s train/validation structure with images and labels.
Extracts your dataset ZIP into a temporary folder.
Matches images with their YOLO .txt labels.
Splits dataset into 80% training, 20% validation.
Copies images and labels into the correct YOLO folders.
Generates data.yaml to tell YOLO where the data is and what classes exist.
Trains YOLOv8 model and saves results in /runs/train_fast/.
Checks accuracy by reading final mAP@0.5 from results CSV.
Packages the model, data.yaml, and training plot into a ZIP file.

Summary:
This repo simplifies fall detection training with YOLOv8. Just add your dataset ZIP, run the script, and get a trained model with minimal manual setup.
