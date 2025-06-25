#!/bin/bash

echo "🚀 SETUP COMPLETO ONNX CON MODELOS REALES"
echo "========================================"
echo "📋 Este script descarga modelos reales y configura todo el sistema"
echo ""

# Verificar directorio
if [ ! -f "run_ocr_fast.sh" ]; then
    echo "❌ Error: Ejecuta desde el directorio del proyecto OCR"
    exit 1
fi

echo "✅ Directorio correcto detectado"
echo ""

# Paso 1: Activar entorno virtual
echo "🐍 1. Activando entorno virtual..."
source venv/bin/activate

# Paso 2: Instalar ONNX Runtime
echo "📦 2. Instalando ONNX Runtime..."
pip install onnxruntime==1.12.1 --quiet
pip install requests --quiet

# Paso 3: Crear estructura
echo "📁 3. Creando estructura de directorios..."
mkdir -p onnx/models

# Paso 4: Descargar modelos reales
echo "📥 4. Descargando modelos ONNX reales..."
chmod +x download_onnx_models.sh
./download_onnx_models.sh

# Paso 5: Actualizar procesadores
echo "🔄 5. Actualizando procesadores ONNX..."
chmod +x update_onnx_processors.sh
./update_onnx_processors.sh

# Paso 6: Crear script de prueba
echo "🧪 6. Creando script de prueba..."
cat > test_onnx_real.sh << 'EOF'
#!/bin/bash

echo "🧪 PROBANDO OCR ONNX CON MODELOS REALES"
echo "======================================"

IMAGE_PATH="/home/userx/tmp/20250620-A_214056942235719@lid_Juanc_12-59.png"

if [ "$1" != "" ]; then
    IMAGE_PATH="$1"
fi

if [ ! -f "$IMAGE_PATH" ]; then
    echo "❌ Imagen no encontrada: $IMAGE_PATH"
    echo "   Uso: ./test_onnx_real.sh /ruta/a/imagen.png"
    exit 1
fi

echo "📸 Imagen de prueba: $IMAGE_PATH"
echo ""

echo "🔍 Verificando modelos ONNX..."
ls -la onnx/models/

echo ""
echo "⚡ Ejecutando OCR ONNX Real:"
time ./run_onnx_real.sh "$IMAGE_PATH" --verbose

echo ""
echo "📊 Comparando con Tesseract:"
time ./run_ocr_fast.sh "$IMAGE_PATH" --compact > /tmp/tesseract_comparison.json

echo ""
echo "✅ Prueba completada"
echo "📁 Resultados guardados para comparación"
EOF

chmod +x test_onnx_real.sh

echo ""
echo "✅ SETUP COMPLETO ONNX TERMINADO"
echo ""
echo "📁 Sistema completo creado:"
echo "   ✅ Modelos ONNX reales descargados"
echo "   ✅ Procesadores ONNX actualizados"
echo "   ✅ Scripts ejecutables creados"
echo "   ✅ Sistema de clasificación implementado"
echo ""
echo "🚀 COMANDOS DISPONIBLES:"
echo "   ./run_onnx_real.sh imagen.png        → OCR con modelos ONNX reales"
echo "   ./test_onnx_real.sh imagen.png       → Probar sistema completo"
echo "   ./compare_all_versions.sh imagen.png → Comparar todas las versiones"
echo ""
echo "🎯 PRÓXIMO PASO:"
echo "   ./test_onnx_real.sh \"/home/userx/tmp/20250620-A_214056942235719@lid_Juanc_12-59.png\""
