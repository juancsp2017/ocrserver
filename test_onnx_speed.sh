#!/bin/bash

echo "⚡ COMPARACIÓN DE VELOCIDAD OCR"
echo "=============================="

IMAGE_PATH="/home/userx/tmp/20250620-A_214056942235719@lid_Juanc_12-59.png"

if [ ! -f "$IMAGE_PATH" ]; then
    echo "❌ Imagen de prueba no encontrada: $IMAGE_PATH"
    echo "   Cambia la ruta en el script"
    exit 1
fi

echo "📸 Imagen de prueba: $IMAGE_PATH"
echo ""

echo "🐌 Tesseract Rápido:"
time ./run_ocr_fast.sh "$IMAGE_PATH" --compact > /tmp/tesseract_result.json
echo "   Resultado guardado en: /tmp/tesseract_result.json"
echo ""

echo "⚡ ONNX Híbrido:"
time ./run_onnx_hybrid.sh "$IMAGE_PATH" --compact > /tmp/onnx_result.json
echo "   Resultado guardado en: /tmp/onnx_result.json"
echo ""

echo "📊 COMPARAR RESULTADOS:"
echo "   cat /tmp/tesseract_result.json"
echo "   cat /tmp/onnx_result.json"
