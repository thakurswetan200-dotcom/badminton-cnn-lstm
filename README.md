\# 🏸 Badminton CNN-LSTM



An AI-powered badminton stroke recognition system that uses a CNN-LSTM deep learning model to classify badminton shots from video.



The project combines a React frontend, FastAPI backend, OpenCV video processing, and a PyTorch CNN-LSTM model to provide an end-to-end video-based shot prediction system.



\---



\## 📌 Overview



Badminton involves fast and complex movements where identifying individual strokes from match footage can be difficult to perform manually.



This project aims to automate badminton stroke recognition by analyzing video frames and classifying the detected stroke into one of five categories:



\- Clear

\- Drive

\- Drop Shot

\- Net Shot

\- Smash



The system accepts a badminton video through a web interface, processes the video, extracts a sequence of frames, and passes the frames through a trained CNN-LSTM model.



The final prediction and confidence score are returned to the React frontend.



\---



\## 🎯 Objective



The main objective of this project is to build an automated badminton stroke recognition system that can:



\- Accept badminton video as input

\- Extract representative frames from the video

\- Learn spatial information from individual frames

\- Learn temporal information across consecutive frames

\- Classify the badminton stroke

\- Return the predicted class and confidence

\- Provide the result through a web-based interface



\---



\## 🧠 System Architecture



> An architecture diagram will be added here.



The overall system follows this pipeline:



```text

User

&#x20; ↓

React Frontend

&#x20; ↓

HTTP POST Request

&#x20; ↓

FastAPI Backend

&#x20; ↓

Video Upload

&#x20; ↓

predict.py

&#x20; ↓

OpenCV Video Processing

&#x20; ↓

16 Sampled Frames

&#x20; ↓

CNN Spatial Feature Extraction

&#x20; ↓

LSTM Temporal Modeling

&#x20; ↓

5-Class Classification

&#x20; ↓

Softmax Probabilities

&#x20; ↓

JSON Response

&#x20; ↓

React Frontend

&#x20; ↓

Prediction + Confidence

