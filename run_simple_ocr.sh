#!/bin/bash

# Script para ejecutar OCR Simple
cd "$(dirname "$0")"

# Variables de entorno para CPUs antiguas
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Ejecutar OCR
python3 simple_ocr_processor.py "$@"
