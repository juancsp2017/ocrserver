#!/bin/bash

# Script para OCR Híbrido ONNX + Tesseract
cd "$(dirname "$0")"
source venv/bin/activate

# Variables de entorno optimizadas
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Configurar ONNX Runtime para CPU
export ORT_DISABLE_ALL_OPTIMIZATIONS=0
export ORT_ENABLE_CPU_FP16_OPS=0

# Ejecutar OCR híbrido
python3 onnx/ocr_onnx_hybrid.py "$@"
