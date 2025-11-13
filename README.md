---
title: Deepfake ECG Generator - Plus
emoji: 👁
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.48.0
app_file: app.py
pinned: true
license: cc-by-4.0
short_description: Fake ECG Generator
---

# Deepfake ECG Generator GUI

Allows to generate ECGs. Based on the following paper:

https://www.nature.com/articles/s41598-021-01295-2

# Run locally

## Prepare venv and install dependencies
```bash
mkdir -p ~/python-environments/deepfake-ecg
python3 -m venv ~/python-environments/deepfake-ecg
. ~/python-environments/deepfake-ecg/bin/activate
pip install -r requirements.txt
```

## Run the application
```bash
./app.py
```
Then, connect a web browser to [http://127.0.0.1:7860/](http://127.0.0.1:7860/) to use the application