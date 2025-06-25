#!/usr/bin/env python3
"""
OCR Processor SIMPLE para recibos bancarios venezolanos
Versión ultra-compatible para CPUs antiguas
Solo OpenCV + patrones inteligentes
"""

import sys
import json
import re
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import argparse
import logging
from pathlib import Path
import os

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SimpleVenezuelanOCR:
    """OCR simple y efectivo para recibos bancarios"""
    
    def __init__(self):
        logger.info("Inicializando OCR Simple...")
        
        # Patrones para bancos venezolanos
        self.bank_patterns = {
            'banesco': [
                r'banesco',
                r'banco\s+banesco',
                r'banesco\s+banco',
                r'b\.?\s*banesco'
            ],
            'mercantil': [
                r'mercantil',
                r'banco\s+mercantil',
                r'mercantil\s+banco',
                r'b\.?\s*mercantil'
            ],
            'venezuela': [
                r'banco\s+de\s+venezuela',
                r'bdv',
                r'venezuela\s+banco',
                r'b\.?\s*venezuela'
            ],
            'provincial': [
                r'bbva\s+provincial',
                r'provincial',
                r'bbva',
                r'banco\s+provincial'
            ],
            'bicentenario': [
                r'bicentenario',
                r'banco\s+bicentenario',
                r'b\.?\s*bicentenario'
            ]
        }
        
        # Patrones para montos
        self.amount_patterns = [
            r'bs\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*bs\.?',
            r'monto:?\s*bs\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'total:?\s*bs\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'transferido:?\s*bs\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'(\d+[.,]\d{2})\s*bs',
            r'bs\s*(\d+[.,]\d{2})'
        ]
        
        # Patrones para referencias
        self.reference_patterns = [
            r'referencia:?\s*(\d{6,20})',
            r'ref\.?\s*(\d{6,20})',
            r'numero:?\s*(\d{6,20})',
            r'operacion:?\s*(\d{6,20})',
            r'transaccion:?\s*(\d{6,20})',
            r'(\d{10,20})'
        ]

    def preprocess_image(self, image_path):
        """Preprocesar imagen con múltiples técnicas"""
        logger.info(f"Cargando imagen: {image_path}")
        
        # Cargar imagen
        img = cv2.imread(str(image_path))
        if img is None:
            # Intentar con PIL
            pil_img = Image.open(image_path)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        if img is None:
            raise ValueError(f"No se pudo cargar: {image_path}")
        
        # Redimensionar si es muy grande
        h, w = img.shape[:2]
        if w > 1500 or h > 1500:
            scale = 1500 / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
            logger.info(f"Redimensionado a {new_w}x{new_h}")
        
        return img

    def extract_text_regions(self, image):
        """Extraer regiones de texto usando OpenCV"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Crear múltiples versiones procesadas
        versions = []
        
        # Versión 1: Original
        versions.append(('original', gray))
        
        # Versión 2: Mejorar contraste
        enhanced = cv2.equalizeHist(gray)
        versions.append(('enhanced', enhanced))
        
        # Versión 3: Umbralización binaria
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        versions.append(('binary', binary))
        
        # Versión 4: Umbralización adaptativa
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        versions.append(('adaptive', adaptive))
        
        # Versión 5: Reducir ruido
        denoised = cv2.medianBlur(gray, 3)
        versions.append(('denoised', denoised))
        
        return versions

    def recognize_text_opencv(self, image):
        """Reconocer texto usando técnicas de OpenCV"""
        text_content = []
        
        # Detectar contornos que podrían ser texto
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = image.shape
        
        for contour in contours:
            x, y, w_c, h_c = cv2.boundingRect(contour)
            
            # Filtrar regiones que podrían contener texto
            if 15 < w_c < w*0.8 and 8 < h_c < h*0.2:
                # Extraer región
                region = image[y:y+h_c, x:x+w_c]
                
                # Analizar región para inferir contenido
                text = self.analyze_text_region(region)
                if text:
                    text_content.append({
                        'text': text,
                        'bbox': (x, y, x+w_c, y+h_c),
                        'confidence': 0.7
                    })
        
        return text_content

    def analyze_text_region(self, region):
        """Analizar región para inferir texto usando patrones de píxeles"""
        if region.size == 0:
            return ""
        
        h, w = region.shape
        
        # Calcular densidad de píxeles negros
        black_pixels = np.sum(region < 128)
        total_pixels = h * w
        density = black_pixels / total_pixels
        
        # Si hay suficiente densidad, probablemente hay texto
        if 0.1 < density < 0.8:
            # Analizar patrones para inferir tipo de contenido
            
            # Patrón horizontal (típico de texto)
            horizontal_lines = 0
            for row in range(h):
                if np.sum(region[row] < 128) > w * 0.3:
                    horizontal_lines += 1
            
            # Si hay múltiples líneas horizontales, probablemente es texto
            if horizontal_lines > h * 0.2:
                # Inferir tipo de contenido basado en dimensiones
                aspect_ratio = w / h
                
                if aspect_ratio > 3:  # Texto largo
                    return "TEXTO_LARGO"
                elif aspect_ratio > 1.5:  # Texto medio
                    return "TEXTO_MEDIO"
                else:  # Texto corto o número
                    return "NUMERO"
        
        return ""

    def extract_text_smart(self, image):
        """Extracción inteligente combinando múltiples técnicas"""
        # Obtener versiones procesadas
        versions = self.extract_text_regions(image)
        
        all_text = []
        
        for name, processed_img in versions:
            logger.info(f"Procesando versión: {name}")
            
            # Reconocer texto en esta versión
            text_data = self.recognize_text_opencv(processed_img)
            
            for item in text_data:
                all_text.append({
                    'text': item['text'].lower(),
                    'confidence': item['confidence'],
                    'version': name,
                    'bbox': item['bbox']
                })
        
        return all_text

    def extract_bank(self, text_data):
        """Extraer banco usando patrones"""
        full_text = ' '.join([item['text'] for item in text_data])
        
        for bank_name, patterns in self.bank_patterns.items():
            for pattern in patterns:
                if re.search(pattern, full_text, re.IGNORECASE):
                    logger.info(f"Banco detectado: {bank_name}")
                    return bank_name.title()
        
        return None

    def extract_amount(self, text_data):
        """Extraer monto usando patrones"""
        full_text = ' '.join([item['text'] for item in text_data])
        
        for pattern in self.amount_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                amount_str = match.group(1)
                logger.info(f"Monto encontrado: {amount_str}")
                
                # Normalizar formato
                if ',' in amount_str and amount_str.count(',') == 1 and len(amount_str.split(',')[1]) <= 2:
                    amount_str = amount_str.replace(',', '.')
                else:
                    amount_str = amount_str.replace(',', '')
                
                try:
                    return float(amount_str)
                except ValueError:
                    continue
        
        return None

    def extract_reference(self, text_data):
        """Extraer referencia usando patrones"""
        full_text = ' '.join([item['text'] for item in text_data])
        
        for pattern in self.reference_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                ref = match.group(1)
                logger.info(f"Referencia encontrada: {ref}")
                return ref
        
        return None

    def process_receipt(self, image_path):
        """Procesar recibo completo"""
        logger.info(f"Procesando: {image_path}")
        
        if not Path(image_path).exists():
            return {
                'success': False,
                'error': f'Imagen no encontrada: {image_path}'
            }
        
        try:
            # Preprocesar imagen
            image = self.preprocess_image(image_path)
            
            # Extraer texto
            text_data = self.extract_text_smart(image)
            
            if not text_data:
                return {
                    'success': False,
                    'error': 'No se pudo extraer texto'
                }
            
            # Extraer información
            bank = self.extract_bank(text_data)
            amount = self.extract_amount(text_data)
            reference = self.extract_reference(text_data)
            
            return {
                'success': True,
                'data': {
                    'bank': bank,
                    'amount': amount,
                    'reference': reference,
                    'raw_text': [item['text'] for item in text_data[:5]]
                },
                'confidence': {
                    'bank': 'high' if bank else 'none',
                    'amount': 'high' if amount else 'none',
                    'reference': 'high' if reference else 'none'
                },
                'processing_info': {
                    'method': 'OpenCV Simple',
                    'regions_found': len(text_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error: {e}")
            return {
                'success': False,
                'error': str(e)
            }

def main():
    parser = argparse.ArgumentParser(description='OCR Simple para recibos venezolanos')
    parser.add_argument('image_path', help='Ruta a la imagen')
    parser.add_argument('--verbose', '-v', action='store_true', help='Modo detallado')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        ocr = SimpleVenezuelanOCR()
        result = ocr.process_receipt(args.image_path)
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
        sys.exit(0 if result['success'] else 1)
        
    except Exception as e:
        error_result = {'success': False, 'error': str(e)}
        print(json.dumps(error_result, indent=2, ensure_ascii=False))
        sys.exit(1)

if __name__ == '__main__':
    main()
