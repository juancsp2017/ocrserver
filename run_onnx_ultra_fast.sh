#!/bin/bash

# Script para OCR ONNX Ultra Rápido (2-5 segundos)
cd "$(dirname "$0")"
source venv/bin/activate

# Variables de entorno ultra optimizadas
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ONNX ultra rápido
export ORT_DISABLE_ALL_OPTIMIZATIONS=1
export ORT_ENABLE_CPU_FP16_OPS=0

# Ejecutar OCR ultra rápido
python3 onnx/ocr_onnx_ultra_fast.py "$@"
