# Plastic Waste Detection and Segmentation using YOLO

This project is designed to detect and segment plastic waste in video footage using a YOLO (You Only Look Once) model. The model is trained to identify plastic waste objects, draw bounding boxes around them, and apply segmentation masks to highlight the detected areas. The project also logs detection events for further analysis.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)

## Introduction

Plastic waste is a significant environmental issue, and automated detection systems can help in monitoring and managing waste more effectively. This project leverages the power of YOLO, a state-of-the-art object detection model, to detect and segment plastic waste in video footage. The project also includes a logging mechanism to record detection events for further analysis.

## Features

- **Object Detection**: Detects plastic waste objects in video frames.
- **Segmentation Masks**: Applies segmentation masks to highlight detected objects.
- **Tracking**: Tracks objects across frames using unique IDs.
- **Logging**: Logs detection events with timestamps, object IDs, class names, and bounding box coordinates.
- **Real-time Processing**: Processes video frames in real-time and displays the results.
