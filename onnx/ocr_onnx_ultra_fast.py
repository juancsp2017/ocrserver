#!/usr/bin/env python3
"""
OCR ONNX ULTRA RÁPIDO - Versión extremadamente optimizada
Objetivo: 2-5 segundos de procesamiento
"""

import sys
import json
import re
import cv2
import numpy as np
from PIL import Image
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Logging mínimo para velocidad
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

class UltraFastOCR:
    """OCR ultra rápido con ONNX optimizado al máximo"""
    
    def __init__(self):
        self.onnx_session = None
        self.method = 'Tesseract'
        
        if ONNX_AVAILABLE:
            self._setup_onnx_ultra_fast()
        
        # Patrones ultra optimizados (solo los más efectivos)
        self.patterns = {
            'bank': {
                'mercantil': r'mercantil|0105',
                'venezuela': r'bdv|pagom[oó]vilbdv|0102',
                'banesco': r'banesco|0134',
                'provincial': r'provincial|0108'
            },
            'amount': [
                r'(\d{1,6}[.,]\d{2})\s*bs',
                r'bs\.?\s*(\d{1,6}[.,]\d{2})'
            ],
            'reference': [
                r'operaci[oó]n:?\s*(\d{8,20})',
                r'(\d{10,20})'
            ],
            'date': r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',
            'time': r'(\d{1,2}:\d{2})',
            'id': r'identificaci[oó]n:?\s*(\d{6,9})',
            'origin': r'origen:?\s*(\d{4}\*{2,4}\d{2,4})',
            'destination': r'destino:?\s*(\d{10,11})',
            'bank_code': r'banco:?\s*(\d{4})',
            'phone': r'\b(\d{11})\b'
        }

    def _setup_onnx_ultra_fast(self):
        """Configurar ONNX para máxima velocidad"""
        try:
            # Configuración ultra agresiva para velocidad
            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = 1
            sess_options.intra_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            sess_options.enable_cpu_mem_arena = False
            sess_options.enable_mem_pattern = False
            
            self.method = 'ONNX_Ultra'
            logger.info("ONNX Ultra Fast configurado")
            
        except Exception as e:
            logger.warning(f"Error ONNX setup: {e}")

    def preprocess_ultra_fast(self, image_path):
        """Preprocesamiento ultra rápido"""
        img = cv2.imread(str(image_path))
        if img is None:
            img = cv2.cvtColor(np.array(Image.open(image_path)), cv2.COLOR_RGB2BGR)
        
        # Redimensionado agresivo para velocidad máxima
        h, w = img.shape[:2]
        if w > 800 or h > 800:
            scale = 800 / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Solo una versión optimizada
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.equalizeHist(gray)
        
        return enhanced

    def extract_text_ultra_fast(self, image_path):
        """Extracción ultra rápida"""
        try:
            processed_img = self.preprocess_ultra_fast(image_path)
            
            # Solo una pasada de Tesseract con configuración ultra rápida
            text = pytesseract.image_to_string(
                processed_img, 
                config='--oem 3 --psm 6 -l spa'  # Solo español para velocidad
            )
            
            return text.lower().strip()
            
        except Exception as e:
            logger.error(f"Error extracción: {e}")
            return ""

    def extract_all_data_ultra_fast(self, text):
        """Extracción de datos ultra optimizada"""
        data = {}
        
        # Banco (una sola pasada)
        for bank, pattern in self.patterns['bank'].items():
            if re.search(pattern, text, re.IGNORECASE):
                data['bank'] = bank.title()
                break
        
        # Monto (solo patrones más efectivos)
        for pattern in self.patterns['amount']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    data['amount'] = float(match.group(1).replace(',', '.'))
                    break
                except:
                    continue
        
        # Referencia
        for pattern in self.patterns['reference']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['reference'] = match.group(1)
                break
        
        # Datos adicionales (una sola pasada cada uno)
        patterns_single = [
            ('date', self.patterns['date']),
            ('time', self.patterns['time']),
            ('identification', self.patterns['id']),
            ('origin_account', self.patterns['origin']),
            ('destination_account', self.patterns['destination']),
            ('destination_bank_code', self.patterns['bank_code'])
        ]
        
        for key, pattern in patterns_single:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data[key] = match.group(1)
        
        # Tipo operación
        if 'pago' in text and 'móvil' in text:
            data['operation_type'] = 'pago_movil'
        elif 'transferencia' in text:
            data['operation_type'] = 'transferencia'
        
        # Teléfonos
        phones = re.findall(self.patterns['phone'], text)
        if phones:
            data['phone_numbers'] = list(set(phones))
        
        return data

    def process_ultra_fast(self, image_path):
        """Procesamiento ultra rápido completo"""
        if not Path(image_path).exists():
            return {'success': False, 'error': f'Imagen no encontrada: {image_path}'}
        
        try:
            start_time = datetime.now()
            
            # Extraer texto
            text = self.extract_text_ultra_fast(image_path)
            
            if not text:
                return {'success': False, 'error': 'No se pudo extraer texto'}
            
            # Extraer datos
            data = self.extract_all_data_ultra_fast(text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'data': data,
                'confidence': {
                    'bank': 'high' if data.get('bank') else 'none',
                    'amount': 'high' if data.get('amount') else 'none',
                    'reference': 'high' if data.get('reference') else 'none'
                },
                'processing_info': {
                    'method': self.method,
                    'processing_time': f"{processing_time:.2f}s",
                    'extraction_timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description='OCR ONNX Ultra Rápido')
    parser.add_argument('image_path', help='Ruta a la imagen')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--compact', '-c', action='store_true')
    
    args = parser.parse_args()
    
    try:
        ocr = UltraFastOCR()
        result = ocr.process_ultra_fast(args.image_path)
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
        sys.exit(0 if result['success'] else 1)
        
    except Exception as e:
        print(json.dumps({'success': False, 'error': str(e)}, indent=2, ensure_ascii=False))
        sys.exit(1)

if __name__ == '__main__':
    main()
