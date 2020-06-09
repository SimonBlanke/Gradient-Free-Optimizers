#!/bin/bash

# Run Tests
pytest \
    -p no:warnings \
    tests/ \
    --durations=10 \
    -rfEX
