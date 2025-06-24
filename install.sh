#!/bin/bash

# Script de instalación OPTIMIZADO para CPUs antiguas
# Sin PyTorch - Solo OpenCV y bibliotecas compatibles

set -e

echo "🚀 Instalando OCR para recibos bancarios venezolanos (Versión CPU Antigua)..."

# Verificar recursos del sistema
echo "📊 Verificando recursos del sistema..."
TOTAL_RAM=$(free -m | awk 'NR==2{printf "%.0f", $2/1024}')
AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')

echo "💾 RAM disponible: ${TOTAL_RAM}GB"
echo "💿 Espacio disponible: ${AVAILABLE_SPACE}GB"

if [ "$TOTAL_RAM" -lt 3 ]; then
    echo "⚠️  Advertencia: RAM disponible ($TOTAL_RAM GB) puede ser insuficiente"
fi

if [ "$AVAILABLE_SPACE" -lt 2 ]; then
    echo "❌ Error: Espacio insuficiente ($AVAILABLE_SPACE GB). Se requieren al menos 2GB"
    exit 1
fi

# Detectar arquitectura de CPU
echo "🔍 Detectando arquitectura de CPU..."
CPU_INFO=$(cat /proc/cpuinfo | grep -E "model name|flags" | head -2)
echo "CPU: $CPU_INFO"

# Actualizar sistema
echo "🔄 Actualizando sistema..."
sudo apt update

# Instalar dependencias del sistema (versiones específicas para compatibilidad)
echo "📦 Instalando dependencias del sistema..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    libopencv-dev \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev

# Crear directorio del proyecto
PROJECT_DIR="$HOME/venezuelan-bank-ocr"
echo "📁 Creando directorio del proyecto en $PROJECT_DIR..."
rm -rf "$PROJECT_DIR" 2>/dev/null || true
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Crear entorno virtual
echo "🐍 Creando entorno virtual Python..."
python3 -m venv venv
source venv/bin/activate

# Actualizar pip y setuptools
echo "⬆️  Actualizando herramientas base..."
pip install --upgrade pip setuptools wheel

# Instalar NumPy compatible con CPUs antiguas
echo "🔢 Instalando NumPy compatible..."
pip install "numpy<1.25.0" --no-binary numpy

# Instalar OpenCV sin optimizaciones AVX
echo "👁️  Instalando OpenCV compatible..."
pip install opencv-python-headless==4.5.5.64

# Instalar Pillow
echo "🖼️  Instalando Pillow..."
pip install Pillow==9.5.0

# Instalar Tesseract Python (alternativa a EasyOCR)
echo "📝 Instalando pytesseract..."
sudo apt install -y tesseract-ocr tesseract-ocr-spa
pip install pytesseract==0.3.10

# Instalar dependencias adicionales
echo "📚 Instalando dependencias adicionales..."
pip install scikit-image==0.19.3 --no-binary scikit-image

# Verificar instalación
echo "✅ Verificando instalación..."
python3 -c "
import cv2
import numpy as np
import PIL
import pytesseract
print('✅ OpenCV version:', cv2.__version__)
print('✅ NumPy version:', np.__version__)
print('✅ PIL version:', PIL.__version__)
print('✅ Tesseract version:', pytesseract.get_tesseract_version())
print('✅ Todas las dependencias instaladas correctamente')
"

# Crear script ejecutable
echo "🔧 Creando script ejecutable..."
cat > run_ocr.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

# Variables de entorno para CPUs antiguas
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 ocr_processor.py "$@"
EOF

chmod +x run_ocr.sh

# Crear script de diagnóstico
cat > diagnose.sh << 'EOF'
#!/bin/bash
echo "🔍 Diagnóstico del sistema OCR"
echo "================================"

cd "$(dirname "$0")"
source venv/bin/activate

echo "📊 Información del sistema:"
echo "CPU: $(cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d: -f2)"
echo "RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "Espacio: $(df -h . | tail -1 | awk '{print $4}')"

echo ""
echo "📦 Versiones instaladas:"
python3 -c "
import sys
print('Python:', sys.version)
try:
    import cv2
    print('OpenCV:', cv2.__version__)
except: print('OpenCV: ERROR')

try:
    import numpy as np
    print('NumPy:', np.__version__)
except: print('NumPy: ERROR')

try:
    import PIL
    print('PIL:', PIL.__version__)
except: print('PIL: ERROR')

try:
    import pytesseract
    print('Tesseract:', pytesseract.get_tesseract_version())
except: print('Tesseract: ERROR')
"

echo ""
echo "🧪 Prueba básica:"
python3 -c "
import cv2
import numpy as np
print('✅ Creando imagen de prueba...')
img = np.zeros((100, 300, 3), dtype=np.uint8)
cv2.putText(img, 'BANCO MERCANTIL BS. 1,500.00', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.imwrite('test_image.png', img)
print('✅ Imagen de prueba creada: test_image.png')
"
EOF

chmod +x diagnose.sh

echo ""
echo "🎉 ¡Instalación completada!"
echo ""
echo "📍 Ubicación: $PROJECT_DIR"
echo "🔍 Diagnóstico: ./diagnose.sh"
echo "🚀 Uso: ./run_ocr.sh /ruta/a/imagen.png"
echo "📖 Ver README.md para más información"
echo ""
echo "⚠️  IMPORTANTE: Esta versión usa Tesseract en lugar de EasyOCR"
echo "   para mayor compatibilidad con CPUs antiguas"
