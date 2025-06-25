#!/bin/bash

echo "🔍 Diagnóstico OCR Simple"
echo "========================="

cd "$(dirname "$0")"

echo "📊 Sistema:"
echo "CPU: $(cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d: -f2)"
echo "RAM: $(free -h | grep Mem | awk '{print $2}')"

echo ""
echo "📦 Python y librerías:"
python3 -c "
import sys
print('Python:', sys.version.split()[0])

try:
    import cv2
    print('✅ OpenCV:', cv2.__version__)
except Exception as e:
    print('❌ OpenCV:', e)

try:
    import numpy as np
    print('✅ NumPy:', np.__version__)
except Exception as e:
    print('❌ NumPy:', e)

try:
    import PIL
    print('✅ PIL:', PIL.__version__)
except Exception as e:
    print('❌ PIL:', e)
"

echo ""
echo "🧪 Prueba básica:"
python3 test_simple_ocr.py
