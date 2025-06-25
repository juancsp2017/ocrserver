#!/bin/bash

echo "🚀 INSTALACIÓN FRESCA OCR - OpenCV + ONNX Runtime"
echo "================================================="

# Variables
PROJECT_NAME="ocr-venezolano-fresh"
PROJECT_DIR="/home/userx/$PROJECT_NAME"

echo "📍 Creando proyecto en: $PROJECT_DIR"

# Crear directorio del proyecto
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

echo "📦 Instalando dependencias del sistema..."
# Instalar SOLO las dependencias necesarias
sudo apt install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    curl

echo "🔢 Instalando NumPy compatible..."
# Instalar NumPy compatible con CPUs antiguas
python3 -m pip install --user --upgrade pip
python3 -m pip install --user "numpy==1.21.6" --no-binary numpy

echo "👁️  Instalando OpenCV optimizado..."
# Instalar OpenCV compilado para compatibilidad máxima
python3 -m pip install --user "opencv-python-headless==4.5.5.64"

echo "🖼️  Instalando Pillow..."
python3 -m pip install --user "Pillow==9.5.0"

echo "🧠 Instalando ONNX Runtime CPU..."
# ONNX Runtime específico para CPUs antiguas
python3 -m pip install --user "onnxruntime==1.12.1"

echo "📚 Instalando dependencias adicionales..."
python3 -m pip install --user "requests==2.28.2"
python3 -m pip install --user "tqdm==4.64.1"

echo "📁 Creando estructura de directorios..."
mkdir -p models
mkdir -p temp
mkdir -p test_images

echo "✅ INSTALACIÓN BASE COMPLETADA"
echo "📍 Proyecto creado en: $PROJECT_DIR"
