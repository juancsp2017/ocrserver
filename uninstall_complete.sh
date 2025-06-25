#!/bin/bash

# Script para desinstalar COMPLETAMENTE todas las versiones anteriores
echo "🗑️  Desinstalación completa del sistema OCR..."

# Ir al directorio home
cd "$HOME"

# Desactivar cualquier entorno virtual
if [[ "$VIRTUAL_ENV" != "" ]]; then
    deactivate 2>/dev/null || true
fi

# Eliminar directorios del proyecto
echo "📁 Eliminando directorios del proyecto..."
rm -rf venezuelan-bank-ocr 2>/dev/null || true
rm -rf venezuelan-bank-ocr-* 2>/dev/null || true

# Limpiar cache de pip global
echo "🧹 Limpiando cache de pip..."
python3 -m pip cache purge 2>/dev/null || true
pip3 cache purge 2>/dev/null || true

# Limpiar archivos temporales
rm -rf ~/.cache/pip 2>/dev/null || true
rm -rf /tmp/pip-* 2>/dev/null || true

# Desinstalar paquetes problemáticos globalmente (opcional)
echo "🔄 Limpiando instalaciones globales problemáticas..."
pip3 uninstall torch torchvision easyocr -y 2>/dev/null || true

echo "✅ Desinstalación completa terminada"
echo "🚀 Ahora puedes ejecutar install_onnx.sh"
