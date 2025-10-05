#!/bin/bash

[ "$1" = "-a" ] && source .venv/bin/activate && return

python3 -m venv .venv

source .venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt