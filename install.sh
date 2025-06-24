#!/bin/bash

# Script de instalación para OCR de recibos bancarios venezolanos
# Optimizado para servidores con recursos limitados (4GB RAM, <1GB disco)

set -e

echo "🚀 Instalando OCR para recibos bancarios venezolanos..."

# Verificar recursos del sistema
echo "📊 Verificando recursos del sistema..."
TOTAL_RAM=$(free -m | awk 'NR==2{printf "%.0f", $2/1024}')
AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')

if [ "$TOTAL_RAM" -lt 3 ]; then
    echo "⚠️  Advertencia: RAM disponible ($TOTAL_RAM GB) puede ser insuficiente"
fi

if [ "$AVAILABLE_SPACE" -lt 2 ]; then
    echo "❌ Error: Espacio insuficiente ($AVAILABLE_SPACE GB). Se requieren al menos 2GB"
    exit 1
fi

# Actualizar sistema
echo "🔄 Actualizando sistema..."
sudo apt update

# Instalar dependencias del sistema
echo "📦 Instalando dependencias del sistema..."
sudo apt install -y python3 python3-pip python3-venv libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# Crear directorio del proyecto
PROJECT_DIR="$HOME/venezuelan-bank-ocr"
echo "📁 Creando directorio del proyecto en $PROJECT_DIR..."
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Crear entorno virtual
echo "🐍 Creando entorno virtual Python..."
python3 -m venv venv
source venv/bin/activate

# Actualizar pip
echo "⬆️  Actualizando pip..."
pip install --upgrade pip

# Instalar PyTorch CPU-only (más liviano)
echo "🔥 Instalando PyTorch CPU-only..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Instalar dependencias principales
echo "📚 Instalando dependencias OCR..."
pip install easyocr opencv-python-headless pillow numpy

# Verificar instalación
echo "✅ Verificando instalación..."
python3 -c "import easyocr; import cv2; import PIL; print('✅ Todas las dependencias instaladas correctamente')"

# Crear script ejecutable
echo "🔧 Creando script ejecutable..."
cat > run_ocr.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python3 ocr_processor.py "$@"
EOF

chmod +x run_ocr.sh

echo "🎉 ¡Instalación completada!"
echo ""
echo "📍 Ubicación: $PROJECT_DIR"
echo "🚀 Uso: ./run_ocr.sh /ruta/a/imagen.png"
echo "📖 Ver README.md para más información"
