#!/bin/bash

# Script optimizado para n8n - Extrae TODA la información
cd "$(dirname "$0")"
source venv/bin/activate

# Variables de entorno para CPUs antiguas
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Ejecutar OCR completo
python3 ocr_for_n8n.py "$@"
