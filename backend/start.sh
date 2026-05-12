#!/bin/bash
python streamer.py &
uvicorn main:app --host 0.0.0.0 --port 7860