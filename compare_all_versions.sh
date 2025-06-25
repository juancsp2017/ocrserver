#!/bin/bash

echo "🚀 COMPARACIÓN COMPLETA DE TODAS LAS VERSIONES OCR"
echo "=================================================="

IMAGE_PATH="/home/userx/tmp/20250620-A_214056942235719@lid_Juanc_12-59.png"

if [ "$1" != "" ]; then
    IMAGE_PATH="$1"
fi

if [ ! -f "$IMAGE_PATH" ]; then
    echo "❌ Imagen no encontrada: $IMAGE_PATH"
    echo "   Uso: ./compare_all_versions.sh /ruta/a/imagen.png"
    exit 1
fi

echo "📸 Imagen de prueba: $IMAGE_PATH"
echo ""

# Crear directorio para resultados
mkdir -p /tmp/ocr_comparison
cd /tmp/ocr_comparison

echo "1️⃣  ONNX Ultra Rápido (objetivo: 2-5s):"
time ~/venezuelan-bank-ocr/run_onnx_ultra_fast.sh "$IMAGE_PATH" --compact > ultra_fast_result.json 2>/dev/null
echo "   ✅ Resultado: ultra_fast_result.json"
echo ""

echo "2️⃣  Tesseract Rápido (objetivo: 8-12s):"
time ~/venezuelan-bank-ocr/run_ocr_fast.sh "$IMAGE_PATH" --compact > tesseract_fast_result.json 2>/dev/null
echo "   ✅ Resultado: tesseract_fast_result.json"
echo ""

echo "3️⃣  ONNX Híbrido (objetivo: 3-8s):"
time ~/venezuelan-bank-ocr/run_onnx_hybrid.sh "$IMAGE_PATH" --compact > hybrid_result.json 2>/dev/null
echo "   ✅ Resultado: hybrid_result.json"
echo ""

echo "4️⃣  ONNX Máxima Precisión (objetivo: 8-15s):"
time ~/venezuelan-bank-ocr/run_onnx_precision.sh "$IMAGE_PATH" --compact > precision_result.json 2>/dev/null
echo "   ✅ Resultado: precision_result.json"
echo ""

echo "5️⃣  Tesseract Completo (objetivo: 25-35s):"
time ~/venezuelan-bank-ocr/run_ocr_n8n.sh "$IMAGE_PATH" --compact > tesseract_complete_result.json 2>/dev/null
echo "   ✅ Resultado: tesseract_complete_result.json"
echo ""

echo "📊 ANÁLISIS DE RESULTADOS:"
echo "=========================="

for file in *.json; do
    if [ -f "$file" ]; then
        echo ""
        echo "📄 $file:"
        
        # Extraer datos principales
        bank=$(cat "$file" | jq -r '.data.bank // "N/A"' 2>/dev/null)
        amount=$(cat "$file" | jq -r '.data.amount // "N/A"' 2>/dev/null)
        reference=$(cat "$file" | jq -r '.data.reference // "N/A"' 2>/dev/null)
        time_taken=$(cat "$file" | jq -r '.processing_info.processing_time // "N/A"' 2>/dev/null)
        method=$(cat "$file" | jq -r '.processing_info.method // "N/A"' 2>/dev/null)
        
        echo "   🏦 Banco: $bank"
        echo "   💰 Monto: $amount"
        echo "   🔢 Referencia: $reference"
        echo "   ⏱️  Tiempo: $time_taken"
        echo "   🔧 Método: $method"
    fi
done

echo ""
echo "📁 Todos los resultados guardados en: /tmp/ocr_comparison/"
echo "🔍 Para ver resultado completo: cat /tmp/ocr_comparison/[archivo].json | jq"
