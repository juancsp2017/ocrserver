#!/bin/bash

echo "🗑️  LIMPIEZA COMPLETA - Eliminando TODAS las instalaciones anteriores"
echo "======================================================================"

# Ir al directorio home del usuario
cd /home/userx

echo "📁 Eliminando directorios del proyecto..."
# Eliminar TODOS los directorios relacionados
sudo rm -rf venezuelan-bank-ocr* 2>/dev/null || true
sudo rm -rf ocr-* 2>/dev/null || true
sudo rm -rf *ocr* 2>/dev/null || true

echo "🧹 Limpiando cache de Python..."
# Limpiar cache de pip
python3 -m pip cache purge 2>/dev/null || true
rm -rf ~/.cache/pip 2>/dev/null || true
rm -rf /tmp/pip-* 2>/dev/null || true

echo "🔄 Desinstalando paquetes problemáticos..."
# Desinstalar paquetes que pueden causar conflictos
pip3 uninstall -y torch torchvision easyocr pytesseract opencv-python opencv-python-headless 2>/dev/null || true

echo "📦 Actualizando sistema..."
sudo apt update

echo "✅ LIMPIEZA COMPLETA TERMINADA"
echo "🚀 Ahora puedes proceder con la instalación nueva"
