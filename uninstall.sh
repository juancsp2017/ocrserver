#!/bin/bash

# Script para desinstalar completamente la versión anterior
echo "🗑️  Desinstalando versión anterior..."

# Ir al directorio home
cd "$HOME"

# Desactivar entorno virtual si está activo
if [[ "$VIRTUAL_ENV" != "" ]]; then
    deactivate
fi

# Eliminar directorio completo del proyecto
if [ -d "venezuelan-bank-ocr" ]; then
    echo "📁 Eliminando directorio venezuelan-bank-ocr..."
    rm -rf venezuelan-bank-ocr
fi

# Limpiar cache de pip
echo "🧹 Limpiando cache de pip..."
python3 -m pip cache purge 2>/dev/null || true

echo "✅ Desinstalación completa"
echo "🚀 Ahora puedes ejecutar el nuevo install.sh"
