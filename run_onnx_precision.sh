#!/bin/bash

# Script para OCR ONNX Máxima Precisión (8-15 segundos)
cd "$(dirname "$0")"
source venv/bin/activate

# Variables de entorno para máxima precisión
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OMP_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

# ONNX máxima precisión
export ORT_DISABLE_ALL_OPTIMIZATIONS=0
export ORT_ENABLE_CPU_FP16_OPS=0

# Ejecutar OCR de precisión
python3 onnx/ocr_onnx_precision.py "$@"
