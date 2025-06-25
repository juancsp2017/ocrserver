#!/bin/bash

# Script para OCR ONNX Real con modelos funcionales
cd "$(dirname "$0")"
source venv/bin/activate

# Variables de entorno optimizadas
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Configurar ONNX Runtime
export ORT_DISABLE_ALL_OPTIMIZATIONS=0
export ORT_ENABLE_CPU_FP16_OPS=0

# Ejecutar OCR con modelos reales
python3 onnx/ocr_onnx_real.py "$@"
