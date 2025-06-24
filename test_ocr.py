#!/usr/bin/env python3
"""
Script de prueba para el OCR de recibos bancarios
Versión optimizada para CPUs antiguas
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from ocr_processor import VenezuelanBankOCR

def create_test_image():
    """Crear imagen de prueba con texto bancario"""
    print("🖼️  Creando imagen de prueba...")
    
    # Crear imagen en blanco
    img = np.ones((400, 800, 3), dtype=np.uint8) * 255
    
    # Agregar texto simulando un recibo
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    texts = [
        ("BANCO MERCANTIL", (50, 50), 1.2, (0, 0, 0), 2),
        ("Comprobante de Pago", (50, 100), 0.8, (0, 0, 0), 2),
        ("Monto: Bs. 2,500.50", (50, 150), 1.0, (0, 0, 0), 2),
        ("Referencia: 123456789012", (50, 200), 1.0, (0, 0, 0), 2),
        ("Fecha: 21/06/2025", (50, 250), 0.8, (0, 0, 0), 2),
        ("Operacion Exitosa", (50, 300), 0.8, (0, 100, 0), 2)
    ]
    
    for text, pos, scale, color, thickness in texts:
        cv2.putText(img, text, pos, font, scale, color, thickness)
    
    # Guardar imagen
    test_path = "test_receipt.png"
    cv2.imwrite(test_path, img)
    print(f"✅ Imagen de prueba creada: {test_path}")
    
    return test_path

def test_installation():
    """Probar que la instalación funciona correctamente"""
    print("🧪 Probando instalación del OCR...")
    
    try:
        # Verificar imports
        import cv2
        import numpy as np
        import pytesseract
        from PIL import Image
        
        print("✅ Todas las librerías importadas correctamente")
        
        # Verificar Tesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
        
        # Inicializar OCR
        ocr = VenezuelanBankOCR()
        print("✅ OCR inicializado correctamente")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en las pruebas: {e}")
        return False

def test_ocr_processing():
    """Probar el procesamiento OCR completo"""
    print("🔍 Probando procesamiento OCR...")
    
    try:
        # Crear imagen de prueba
        test_image = create_test_image()
        
        # Inicializar OCR
        ocr = VenezuelanBankOCR()
        
        # Procesar imagen
        result = ocr.process_receipt(test_image)
        
        print("📊 Resultado del procesamiento:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Verificar resultados
        if result['success']:
            data = result['data']
            print(f"✅ Banco detectado: {data['bank']}")
            print(f"✅ Monto detectado: {data['amount']}")
            print(f"✅ Referencia detectada: {data['reference']}")
            
            # Limpiar archivo de prueba
            Path(test_image).unlink(missing_ok=True)
            
            return True
        else:
            print(f"❌ Error en procesamiento: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Error en prueba OCR: {e}")
        return False

def main():
    print("🚀 Iniciando pruebas del sistema OCR (Versión CPU Antigua)...")
    print("=" * 60)
    
    # Probar instalación
    if not test_installation():
        print("\n❌ Falló la prueba de instalación")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    
    # Probar procesamiento OCR
    if not test_ocr_processing():
        print("\n❌ Falló la prueba de procesamiento OCR")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 ¡Todas las pruebas pasaron exitosamente!")
    print("✅ El sistema está listo para usar")
    print("\n📖 Uso:")
    print("   ./run_ocr.sh /ruta/a/imagen.png")
    print("   ./run_ocr.sh /ruta/a/imagen.png --verbose")

if __name__ == '__main__':
    main()
