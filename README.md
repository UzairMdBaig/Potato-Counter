# 🥔 Potato Counter with YOLOv11 and SORT Tracking

This project is a real-time potato counting system built using:
- A **YOLOv11 object detection model** custom trained on annotated potato images
- **OpenCV** for video processing
- The **SORT** algorithm to track detected potatoes
- A counting mechanism that increments when tracked potatoes cross a defined virtual line, ensuirng no duplicates counts occure


<img width="1357" height="693" alt="image" src="https://github.com/user-attachments/assets/5c3d1804-1ad6-4e60-8c93-43e5d4a3a142" />


## How It Works

1. **Object Detection**  
   A YOLOv11m model is trained on annotated potato images to detect potatoes in each frame.

2. **Tracking with SORT**  
   Detected bounding boxes are passed into the [SORT algorithm](https://github.com/abewley/sort) (imported from an external repo) to assign consistent IDs to each potato across frames.

3. **Counting Logic** 
   - The center of each tracked bounding box is calculated.
   - If a bounding box's center **crosses the line** in a specific direction, the potato is counted.



## How to run the project:

1. First install the dependencies
```bash
pip install -r requirements.txt
```

2. Run the manual_counter.py file
```bash
python manual_counter.py
```

