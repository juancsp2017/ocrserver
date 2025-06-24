#!/usr/bin/env python3
"""
Script de prueba para el OCR de recibos bancarios
"""

import sys
import json
from pathlib import Path
from ocr_processor import VenezuelanBankOCR

def test_installation():
    """Probar que la instalación funciona correctamente"""
    print("🧪 Probando instalación del OCR...")
    
    try:
        # Inicializar OCR
        ocr = VenezuelanBankOCR()
        print("✅ OCR inicializado correctamente")
        
        # Probar patrones
        test_text = [
            {'text': 'banco mercantil bs. 1,500.00 ref: 123456789', 'confidence': 0.9}
        ]
        
        bank = ocr.extract_bank(test_text)
        amount = ocr.extract_amount(test_text)
        reference = ocr.extract_reference(test_text)
        
        print(f"✅ Extracción de banco: {bank}")
        print(f"✅ Extracción de monto: {amount}")
        print(f"✅ Extracción de referencia: {reference}")
        
        if bank and amount and reference:
            print("🎉 ¡Todas las pruebas pasaron!")
            return True
        else:
            print("⚠️  Algunas extracciones fallaron")
            return False
            
    except Exception as e:
        print(f"❌ Error en las pruebas: {e}")
        return False

def create_sample_data():
    """Crear datos de ejemplo para pruebas"""
    sample_data = {
        'test_cases': [
            {
                'description': 'Transferencia Banesco',
                'expected': {
                    'bank': 'Banesco',
                    'amount': 2500.50,
                    'reference': '987654321'
                }
            },
            {
                'description': 'Pago Mercantil',
                'expected': {
                    'bank': 'Mercantil',
                    'amount': 150.00,
                    'reference': '123456789'
                }
            }
        ]
    }
    
    with open('test_data.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print("📄 Archivo test_data.json creado")

if __name__ == '__main__':
    print("🚀 Iniciando pruebas del sistema OCR...")
    
    # Probar instalación
    if test_installation():
        print("\n✅ Sistema listo para usar")
        create_sample_data()
    else:
        print("\n❌ Hay problemas con la instalación")
        sys.exit(1)
