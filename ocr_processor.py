#!/usr/bin/env python3
"""
OCR Processor para recibos bancarios venezolanos
Versión optimizada para CPUs antiguas usando Tesseract
"""

import sys
import json
import re
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import argparse
import logging
from pathlib import Path
import os

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VenezuelanBankOCR:
    def __init__(self):
        """Inicializar el procesador OCR"""
        logger.info("Inicializando Tesseract OCR...")
        
        # Configurar Tesseract para español
        self.tesseract_config = '--oem 3 --psm 6 -l spa+eng'
        
        # Verificar que Tesseract funciona
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
        except Exception as e:
            logger.error(f"Error inicializando Tesseract: {e}")
            raise
        
        # Patrones para bancos venezolanos (más flexibles)
        self.bank_patterns = {
            'banesco': [
                r'banesco',
                r'banco\s+banesco',
                r'b\.?\s*banesco'
            ],
            'mercantil': [
                r'mercantil',
                r'banco\s+mercantil',
                r'b\.?\s*mercantil'
            ],
            'venezuela': [
                r'banco\s+de\s+venezuela',
                r'bdv',
                r'b\.?\s*venezuela'
            ],
            'provincial': [
                r'bbva\s+provincial',
                r'provincial',
                r'bbva'
            ],
            'bicentenario': [
                r'bicentenario',
                r'banco\s+bicentenario',
                r'b\.?\s*bicentenario'
            ],
            'tesoro': [
                r'banco\s+del\s+tesoro',
                r'tesoro',
                r'b\.?\s*tesoro'
            ]
        }
        
        # Patrones para montos (más flexibles)
        self.amount_patterns = [
            r'bs\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*bs\.?',
            r'monto:?\s*bs\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'total:?\s*bs\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*bolívares?',
            r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*bolivares?',
            r'(\d+[.,]\d{2})\s*bs',
            r'bs\s*(\d+[.,]\d{2})'
        ]
        
        # Patrones para referencias (más flexibles)
        self.reference_patterns = [
            r'referencia:?\s*(\d{6,20})',
            r'ref\.?\s*(\d{6,20})',
            r'operaci[oó]n:?\s*(\d{6,20})',
            r'n[uú]mero:?\s*(\d{6,20})',
            r'comprobante:?\s*(\d{6,20})',
            r'transacci[oó]n:?\s*(\d{6,20})',
            r'(\d{8,20})'  # Fallback para números largos
        ]

    def preprocess_image(self, image_path):
        """Preprocesar imagen para mejorar OCR con Tesseract"""
        logger.info(f"Preprocesando imagen: {image_path}")
        
        try:
            # Leer imagen con OpenCV
            img = cv2.imread(str(image_path))
            if img is None:
                # Intentar con PIL si OpenCV falla
                pil_img = Image.open(image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            if img is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            # Redimensionar si es muy grande (para ahorrar memoria)
            height, width = img.shape[:2]
            if width > 2000 or height > 2000:
                scale = min(2000/width, 2000/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                logger.info(f"Imagen redimensionada a {new_width}x{new_height}")
            
            # Convertir a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Crear múltiples versiones procesadas
            processed_images = []
            
            # Versión 1: Original en escala de grises
            processed_images.append(('original_gray', gray))
            
            # Versión 2: Mejorar contraste
            enhanced = cv2.equalizeHist(gray)
            processed_images.append(('enhanced', enhanced))
            
            # Versión 3: Reducir ruido
            denoised = cv2.medianBlur(gray, 3)
            processed_images.append(('denoised', denoised))
            
            # Versión 4: Umbralización binaria
            _, thresh_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(('binary', thresh_binary))
            
            # Versión 5: Umbralización adaptativa
            thresh_adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            processed_images.append(('adaptive', thresh_adaptive))
            
            return processed_images
            
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {e}")
            return []

    def extract_text_tesseract(self, image, config=None):
        """Extraer texto usando Tesseract"""
        if config is None:
            config = self.tesseract_config
        
        try:
            # Extraer texto
            text = pytesseract.image_to_string(image, config=config)
            
            # Extraer datos con confianza
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            
            # Filtrar texto con confianza > 30
            confident_text = []
            for i, conf in enumerate(data['conf']):
                if int(conf) > 30:
                    word = data['text'][i].strip()
                    if word:
                        confident_text.append({
                            'text': word,
                            'confidence': int(conf) / 100.0
                        })
            
            return {
                'full_text': text.strip(),
                'words': confident_text
            }
            
        except Exception as e:
            logger.warning(f"Error en Tesseract: {e}")
            return {'full_text': '', 'words': []}

    def extract_text(self, image_path):
        """Extraer texto usando múltiples versiones de la imagen"""
        try:
            # Preprocesar imagen
            processed_images = self.preprocess_image(image_path)
            
            if not processed_images:
                return []
            
            all_text_data = []
            
            # Procesar cada versión de la imagen
            for name, img in processed_images:
                logger.info(f"Procesando versión: {name}")
                
                result = self.extract_text_tesseract(img)
                
                if result['full_text']:
                    all_text_data.append({
                        'version': name,
                        'full_text': result['full_text'].lower(),
                        'words': result['words']
                    })
            
            return all_text_data
            
        except Exception as e:
            logger.error(f"Error extrayendo texto: {e}")
            return []

    def extract_bank(self, text_data):
        """Extraer información del banco"""
        # Combinar todo el texto
        full_text = ' '.join([data['full_text'] for data in text_data])
        
        for bank_name, patterns in self.bank_patterns.items():
            for pattern in patterns:
                if re.search(pattern, full_text, re.IGNORECASE):
                    logger.info(f"Banco detectado: {bank_name} con patrón: {pattern}")
                    return bank_name.title()
        
        return None

    def extract_amount(self, text_data):
        """Extraer monto del pago"""
        # Combinar todo el texto
        full_text = ' '.join([data['full_text'] for data in text_data])
        
        for pattern in self.amount_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                amount_str = match.group(1)
                logger.info(f"Monto encontrado: {amount_str} con patrón: {pattern}")
                
                # Normalizar formato
                # Reemplazar comas por puntos para decimales
                if ',' in amount_str and amount_str.count(',') == 1 and len(amount_str.split(',')[1]) <= 2:
                    amount_str = amount_str.replace(',', '.')
                else:
                    # Remover comas de miles
                    amount_str = amount_str.replace(',', '')
                
                try:
                    return float(amount_str)
                except ValueError:
                    continue
        
        return None

    def extract_reference(self, text_data):
        """Extraer número de referencia"""
        # Combinar todo el texto
        full_text = ' '.join([data['full_text'] for data in text_data])
        
        for pattern in self.reference_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                ref = match.group(1)
                logger.info(f"Referencia encontrada: {ref} con patrón: {pattern}")
                return ref
        
        return None

    def process_receipt(self, image_path):
        """Procesar recibo completo"""
        logger.info(f"Procesando recibo: {image_path}")
        
        # Verificar que existe la imagen
        if not Path(image_path).exists():
            return {
                'success': False,
                'error': f'Imagen no encontrada: {image_path}',
                'data': None
            }
        
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
                'raw_text': [data['full_text'][:200] + '...' if len(data['full_text']) > 200 else data['full_text'] for data in text_data[:3]]
            },
            'confidence': {
                'bank': 'high' if bank else 'none',
                'amount': 'high' if amount else 'none',
                'reference': 'high' if reference else 'none'
            },
            'processing_info': {
                'versions_processed': len(text_data),
                'tesseract_version': str(pytesseract.get_tesseract_version())
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
        error_result = {
            'success': False,
            'error': f'Imagen no encontrada: {image_path}'
        }
        print(json.dumps(error_result, indent=2, ensure_ascii=False))
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
        print(json.dumps(error_result, indent=2, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
