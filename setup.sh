#!/bin/bash

if [[ ! -d .env ]]; then
	python3 -m venv .env
fi

source .env/bin/activate

pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -e transformers
pip install -r requirements.txt

