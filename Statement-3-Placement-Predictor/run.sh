#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Starting Backend API on port 8003..."
python3 -m uvicorn src.main:app --host 0.0.0.0 --port 8003