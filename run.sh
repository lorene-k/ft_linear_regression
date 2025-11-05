#!/bin/bash

set -e

program="$1"

[ "$1" = "-a" ] && source .venv/bin/activate && return

if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ".venv" ]; then
    source .venv/bin/activate
    else
        python3 -m venv .venv
        source .venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
    fi
fi

if [ -z "$program" ]; then
    echo "Usage : ./run.sh [option]"
    echo "(options = train, predict, precision, graph, test)"
else
    case "$program" in
        train) python -m src.train ;;
        predict) python -m src.predict ;;
        precision) python -m src.bonus.precision ;;
        graph) python -m src.bonus.render_graph ;;
        test) pytest -v ;;
        *) echo "Invalid argument." ;;
    esac
fi