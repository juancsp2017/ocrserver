# 📦 Guía de Instalación Completa

## Pasos de Instalación

### 1. Preparar el Servidor

\`\`\`bash
# Conectar al servidor
ssh usuario@tu-servidor.com

# Actualizar sistema
sudo apt update && sudo apt upgrade -y
\`\`\`

### 2. Descargar el Proyecto

\`\`\`bash
# Opción A: Clonar desde GitHub
git clone https://github.com/tu-usuario/venezuelan-bank-ocr.git
cd venezuelan-bank-ocr

# Opción B: Crear manualmente
mkdir venezuelan-bank-ocr
cd venezuelan-bank-ocr
# Copiar todos los archivos del proyecto aquí
\`\`\`

### 3. Ejecutar Instalación Automática

\`\`\`bash
# Hacer ejecutable el script
chmod +x install.sh

# Ejecutar instalación
./install.sh
\`\`\`

### 4. Verificar Instalación

\`\`\`bash
# Probar el sistema
python3 test_ocr.py

# Verificar script ejecutable
./run_ocr.sh --help
\`\`\`

## 🔧 Instalación Manual (si falla la automática)

### Paso 1: Dependencias del Sistema

\`\`\`bash
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1
\`\`\`

### Paso 2: Entorno Virtual

\`\`\`bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
\`\`\`

### Paso 3: PyTorch CPU-only

\`\`\`bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
\`\`\`

### Paso 4: Dependencias OCR

\`\`\`bash
pip install easyocr opencv-python-headless pillow numpy
\`\`\`

### Paso 5: Verificar

\`\`\`bash
python3 -c "import easyocr; import cv2; print('OK')"
\`\`\`

## 🚀 Uso con n8n

### Configuración en n8n

1. **Nodo Execute Command**:
   \`\`\`bash
   /home/usuario/venezuelan-bank-ocr/run_ocr.sh /tmp/imagen.png
   \`\`\`

2. **Capturar salida JSON**:
   - La salida estará en `stdout`
   - Errores en `stderr`
   - Código de salida: 0 = éxito, 1 = error

### Ejemplo de Flujo n8n

\`\`\`json
{
  "command": "/home/usuario/venezuelan-bank-ocr/run_ocr.sh",
  "parameters": ["/tmp/{{ $json.filename }}"],
  "options": {
    "cwd": "/home/usuario/venezuelan-bank-ocr"
  }
}
\`\`\`

## 📊 Monitoreo de Recursos

### Verificar Uso de Memoria

\`\`\`bash
# Durante procesamiento
ps aux | grep python3
free -h
\`\`\`

### Verificar Espacio en Disco

\`\`\`bash
du -sh venezuelan-bank-ocr/
df -h
\`\`\`

## 🐛 Solución de Problemas Comunes

### Error: "No module named 'easyocr'"

\`\`\`bash
cd venezuelan-bank-ocr
source venv/bin/activate
pip install -r requirements.txt
\`\`\`

### Error: "CUDA not available"

Es normal en instalación CPU-only. Ignorar este mensaje.

### Error: "Permission denied"

\`\`\`bash
chmod +x run_ocr.sh
chmod +x install.sh
\`\`\`

### Memoria insuficiente

\`\`\`bash
# Limitar threads
export OMP_NUM_THREADS=1
./run_ocr.sh imagen.png
\`\`\`

### Imagen no se procesa

1. Verificar formato (PNG, JPG, JPEG)
2. Verificar permisos de lectura
3. Probar con imagen más pequeña

## 🔄 Actualización

\`\`\`bash
cd venezuelan-bank-ocr
git pull origin main
source venv/bin/activate
pip install -r requirements.txt --upgrade
\`\`\`

## 📝 Logs y Debugging

### Activar modo verbose

\`\`\`bash
./run_ocr.sh imagen.png --verbose
\`\`\`

### Ver logs del sistema

\`\`\`bash
journalctl -f | grep python3
\`\`\`

## 🎯 Optimización de Rendimiento

### Para servidores con poca RAM

\`\`\`bash
# En ~/.bashrc
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
\`\`\`

### Para múltiples procesamiento

\`\`\`bash
# Procesar en lotes
for img in *.png; do
    ./run_ocr.sh "$img" > "resultado_$img.json"
done
