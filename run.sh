#!/bin/bash

program="$1"

if [ -z "$program" ]; then
    echo "Usage : ./run.sh [option]"
    echo "(options = train, predict, precision, graph, test)"
else
    case "$program" in
        train) python -m src.train ;;
        predict) python -m src.predict ;;
        precision) python -m src.bonus.precision ;;
        graph) python -m src.bonus.render_graph ;;
        test) python -m tests.test ;;
        *) echo "Invalid argument." ;;
    esac
fi