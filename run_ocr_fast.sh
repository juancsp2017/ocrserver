#!/bin/bash

# Script RÁPIDO para n8n - Optimizado para servidores con alta carga
cd "$(dirname "$0")"
source venv/bin/activate

# Variables de entorno optimizadas para velocidad
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Limitar memoria de Tesseract
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Ejecutar OCR rápido
python3 ocr_fast_n8n.py "$@"
