#!/usr/bin/env python3
"""
OCR Processor para recibos bancarios venezolanos
Optimizado para recursos limitados y alta precisión
"""

import sys
import json
import re
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import easyocr
import argparse
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VenezuelanBankOCR:
    def __init__(self):
        """Inicializar el procesador OCR"""
        logger.info("Inicializando EasyOCR...")
        # Usar solo CPU para ahorrar recursos
        self.reader = easyocr.Reader(['es', 'en'], gpu=False)
        
        # Patrones para bancos venezolanos
        self.bank_patterns = {
            'banesco': r'banesco|banco\s+banesco',
            'mercantil': r'mercantil|banco\s+mercantil',
            'venezuela': r'banco\s+de\s+venezuela|bdv',
            'provincial': r'bbva\s+provincial|provincial',
            'bicentenario': r'bicentenario|banco\s+bicentenario',
            'tesoro': r'banco\s+del\s+tesoro|tesoro',
            'caroni': r'banco\s+caroni|caroni',
            'sofitasa': r'sofitasa|banco\s+sofitasa',
            'activo': r'banco\s+activo|activo',
            'plaza': r'banco\s+plaza|plaza'
        }
        
        # Patrones para montos (Bolívares)
        self.amount_patterns = [
            r'bs\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*bs\.?',
            r'monto:?\s*bs\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'total:?\s*bs\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*bolívares?'
        ]
        
        # Patrones para referencias
        self.reference_patterns = [
            r'referencia:?\s*(\d{8,20})',
            r'ref\.?\s*(\d{8,20})',
            r'operación:?\s*(\d{8,20})',
            r'número:?\s*(\d{8,20})',
            r'comprobante:?\s*(\d{8,20})',
            r'transacción:?\s*(\d{8,20})'
        ]

    def preprocess_image(self, image_path):
        """Preprocesar imagen para mejorar OCR"""
        logger.info(f"Preprocesando imagen: {image_path}")
        
        # Leer imagen
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Mejorar contraste con CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Reducir ruido
        denoised = cv2.medianBlur(enhanced, 3)
        
        # Umbralización adaptativa
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return [gray, enhanced, thresh]  # Devolver múltiples versiones

    def extract_text(self, image_path):
        """Extraer texto usando EasyOCR"""
        try:
            # Preprocesar imagen
            processed_images = self.preprocess_image(image_path)
            
            all_text = []
            
            # Procesar imagen original y versiones procesadas
            images_to_process = [str(image_path)] + processed_images
            
            for i, img in enumerate(images_to_process):
                logger.info(f"Procesando versión {i+1}/{len(images_to_process)}...")
                
                try:
                    if isinstance(img, str):
                        # Imagen original
                        results = self.reader.readtext(img)
                    else:
                        # Imagen procesada (numpy array)
                        results = self.reader.readtext(img)
                    
                    # Extraer texto con confianza > 0.5
                    for (bbox, text, confidence) in results:
                        if confidence > 0.5:
                            all_text.append({
                                'text': text.lower().strip(),
                                'confidence': confidence,
                                'version': i
                            })
                            
                except Exception as e:
                    logger.warning(f"Error procesando versión {i}: {e}")
                    continue
            
            return all_text
            
        except Exception as e:
            logger.error(f"Error extrayendo texto: {e}")
            return []

    def extract_bank(self, text_data):
        """Extraer información del banco"""
        full_text = ' '.join([item['text'] for item in text_data])
        
        for bank_name, pattern in self.bank_patterns.items():
            if re.search(pattern, full_text, re.IGNORECASE):
                return bank_name.title()
        
        return None

    def extract_amount(self, text_data):
        """Extraer monto del pago"""
        full_text = ' '.join([item['text'] for item in text_data])
        
        for pattern in self.amount_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                amount_str = match.group(1)
                # Normalizar formato (usar punto como decimal)
                amount_str = amount_str.replace(',', '.')
                try:
                    return float(amount_str)
                except ValueError:
                    continue
        
        return None

    def extract_reference(self, text_data):
        """Extraer número de referencia"""
        full_text = ' '.join([item['text'] for item in text_data])
        
        for pattern in self.reference_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Buscar secuencias largas de números como fallback
        numbers = re.findall(r'\d{8,20}', full_text)
        if numbers:
            return numbers[0]
        
        return None

    def process_receipt(self, image_path):
        """Procesar recibo completo"""
        logger.info(f"Procesando recibo: {image_path}")
        
        # Extraer texto
        text_data = self.extract_text(image_path)
        
        if not text_data:
            return {
                'success': False,
                'error': 'No se pudo extraer texto de la imagen',
                'data': None
            }
        
        # Extraer información clave
        bank = self.extract_bank(text_data)
        amount = self.extract_amount(text_data)
        reference = self.extract_reference(text_data)
        
        # Preparar resultado
        result = {
            'success': True,
            'data': {
                'bank': bank,
                'amount': amount,
                'reference': reference,
                'raw_text': [item['text'] for item in text_data[:10]]  # Primeras 10 líneas
            },
            'confidence': {
                'bank': 'high' if bank else 'none',
                'amount': 'high' if amount else 'none',
                'reference': 'high' if reference else 'none'
            }
        }
        
        return result

def main():
    parser = argparse.ArgumentParser(description='OCR para recibos bancarios venezolanos')
    parser.add_argument('image_path', help='Ruta a la imagen del recibo')
    parser.add_argument('--verbose', '-v', action='store_true', help='Modo verbose')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Verificar que existe la imagen
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(json.dumps({
            'success': False,
            'error': f'Imagen no encontrada: {image_path}'
        }), file=sys.stderr)
        sys.exit(1)
    
    try:
        # Procesar recibo
        ocr = VenezuelanBankOCR()
        result = ocr.process_receipt(image_path)
        
        # Imprimir resultado JSON
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Código de salida
        sys.exit(0 if result['success'] else 1)
        
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e)
        }
        print(json.dumps(error_result), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
