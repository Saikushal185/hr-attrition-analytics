#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -f models/attrition_model.joblib ]; then
    echo "Running analytics + training..."
    python3 src/main.py
fi
python3 -m streamlit run dashboard/app.py
