\# Dock Inference 

Real-time dock light fish detection system using raspberry pi 4 streaming over network and OpenCV inference. 



\## System Overview 



* Raspberry Pi streams H264 video via FFmpeg
* Laptop ingests UDP stream 
* OpenCV-based inference processes frames
* Detection results displayed live



\## Files 



* `dock\_inference\_v1.py` – main inference pipeline
* `ffmpeg\_reader.py` – stream ingestion
* `test\_stream.py` – stream testing utility



\## Goals



* Detect fish activity in real time (working)
* Convert detections into structured events with parameters (size, species, date and time, tide, moon phase, temperature, etc.)
* Store events in database 
* Enable downstream (lol) data analysis and modeling



\## Planned Architecture



\### Stage 1 — Detection (Current)

Pi → FFmpeg → UDP → Frame ingestion → Detection



\### Stage 2 — Event Generation

Detection → Structured Event Object

\- Timestamp

\- Bounding box

\- Confidence score

\- Frame metadata

\- Environmental context (future)



\### Stage 3 — Persistent Storage

Events stored in:

\- SQLite (initial) /undecided 

\- PostgreSQL /undecided



\### Stage 4 — Modeling/Analysis/Prediction

\- Activity frequency modeling

\- Time-of-night pattern analysis

\- Seasonal trend detection

\- Species clustering

\- Environmental correlation analysis



