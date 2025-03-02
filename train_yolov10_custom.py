#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for custom YOLOv10 model with CBAM and BiFPN
"""

import torch
from ultralytics import YOLO

def main():
    """Main function to train the custom YOLOv10 model."""
    # Initialize the model from YAML config with task specified
    model = YOLO('/Users/debasishborah/PycharmProjects/yolo_experiment/ultralytics/cfg/models/v8/yolov8-cbam.yaml', task='detect')
    
    # Print model summary
    model.info()
    
    # Train the model on COCO dataset
    results = model.train(
        data='/Users/debasishborah/Desktop/working/data.yaml',        # COCO dataset
        epochs=10,              # Number of epochs
        imgsz=640,               # Image size
        batch=6,                # Batch size
        workers=8,               # Number of workers
        device='mps',                # GPU device (use 'cpu' for CPU)
        patience=50,             # Early stopping patience
        save=True,               # Save checkpoints
        project='yolov10_custom', # Project name
        name='train',            # Run name
        exist_ok=True,           # Overwrite existing run
        pretrained=False,        # Don't use pretrained weights
        optimizer='AdamW',       # Optimizer
        lr0=0.001,               # Initial learning rate
        lrf=0.01,                # Final learning rate ratio
        momentum=0.937,          # SGD momentum/Adam beta1
        weight_decay=0.0005,     # Optimizer weight decay
        warmup_epochs=3.0,       # Warmup epochs
        warmup_momentum=0.8,     # Warmup initial momentum
        warmup_bias_lr=0.1,      # Warmup initial bias lr
        box=7.5,                 # Box loss gain
        cls=0.5,                 # Cls loss gain
        hsv_h=0.015,             # Image HSV-Hue augmentation (fraction)
        hsv_s=0.7,               # Image HSV-Saturation augmentation (fraction)
        hsv_v=0.4,               # Image HSV-Value augmentation (fraction)
        degrees=0.0,             # Image rotation (+/- deg)
        translate=0.1,           # Image translation (+/- fraction)
        scale=0.5,               # Image scale (+/- gain)
        shear=0.0,               # Image shear (+/- deg)
        perspective=0.0,         # Image perspective (+/- fraction)
        flipud=0.0,              # Image flip up-down (probability)
        fliplr=0.5,              # Image flip left-right (probability)
        mosaic=1.0,              # Image mosaic (probability)
        mixup=0.0,               # Image mixup (probability)
        copy_paste=0.0,          # Segment copy-paste (probability)
    )
    
    # Evaluate the model on the validation set
    results = model.val()
    
    # Save the final model
    model.save('yolov8_custom.pt')
    
    print("Model training completed successfully!")

if __name__ == "__main__":
    main() 