#!/usr/bin/env python3
"""
Prueba del OCR Simple
"""

import cv2
import numpy as np
import json
from simple_ocr_processor import SimpleVenezuelanOCR

def create_test_image():
    """Crear imagen de prueba"""
    print("🖼️  Creando imagen de prueba...")
    
    # Imagen blanca
    img = np.ones((600, 1000, 3), dtype=np.uint8) * 255
    
    # Agregar texto bancario
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Texto del banco
    cv2.putText(img, 'BANCO MERCANTIL', (100, 100), font, 1.5, (0, 0, 0), 3)
    
    # Monto
    cv2.putText(img, 'Monto: Bs. 1,250.75', (100, 200), font, 1.2, (0, 0, 0), 2)
    
    # Referencia
    cv2.putText(img, 'Referencia: 987654321098', (100, 300), font, 1.0, (0, 0, 0), 2)
    
    # Estado
    cv2.putText(img, 'Estado: EXITOSA', (100, 400), font, 1.0, (0, 150, 0), 2)
    
    # Guardar
    cv2.imwrite('test_simple.png', img)
    print("✅ Imagen creada: test_simple.png")
    return 'test_simple.png'

def test_ocr():
    """Probar OCR"""
    print("🧪 Probando OCR Simple...")
    
    # Crear imagen de prueba
    test_img = create_test_image()
    
    # Procesar con OCR
    ocr = SimpleVenezuelanOCR()
    result = ocr.process_receipt(test_img)
    
    print("📊 Resultado:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    if result['success']:
        print("✅ Prueba exitosa!")
        return True
    else:
        print("❌ Prueba falló")
        return False

if __name__ == '__main__':
    test_ocr()
