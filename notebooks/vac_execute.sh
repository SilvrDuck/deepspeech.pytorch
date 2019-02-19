#!/bin/bash

python vac01_create_embedder.py || true
python vac02_create_embeddings.py || true
python vac03_mono.py || true
python vac04_mono-ivect.py || true
python vac07_multi_9.py || true
python vac08_multi_5.py || true