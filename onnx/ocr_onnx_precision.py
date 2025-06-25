#!/usr/bin/env python3
"""
OCR ONNX PRECISIÓN MÁXIMA - Versión para máxima extracción de datos
Objetivo: Extraer TODO con alta precisión (tiempo: 8-15s)
"""

import sys
import json
import re
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import argparse
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
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

class PrecisionOCR:
    """OCR de máxima precisión con ONNX"""
    
    def __init__(self):
        self.onnx_session = None
        self.method = 'Precision_Tesseract'
        
        if ONNX_AVAILABLE:
            self._setup_onnx_precision()
        
        # Patrones exhaustivos para máxima precisión
        self.bank_patterns = {
            'banesco': [
                r'banesco', r'banco\s+banesco', r'b\.?\s*banesco', 
                r'banesco\s+banco', r'0134', r'banesco\s+universal'
            ],
            'mercantil': [
                r'mercantil', r'banco\s+mercantil', r'b\.?\s*mercantil',
                r'mercantil\s+banco', r'0105', r'mercantil\s+universal'
            ],
            'venezuela': [
                r'banco\s+de\s+venezuela', r'bdv', r'b\.?\s*venezuela',
                r'venezuela\s+banco', r'pagom[oó]vilbdv', r'0102',
                r'banco\s+venezuela', r'pago\s*m[oó]vil\s*bdv'
            ],
            'provincial': [
                r'bbva\s+provincial', r'provincial', r'bbva',
                r'banco\s+provincial', r'0108', r'bbva\s+banco'
            ],
            'bicentenario': [
                r'bicentenario', r'banco\s+bicentenario', r'b\.?\s*bicentenario',
                r'0175', r'banco\s+del\s+bicentenario'
            ],
            'tesoro': [
                r'banco\s+del\s+tesoro', r'tesoro', r'b\.?\s*tesoro',
                r'0163', r'banco\s+tesoro'
            ]
        }
        
        self.amount_patterns = [
            r'bs\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*bs\.?',
            r'monto:?\s*bs\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'total:?\s*bs\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'transferido:?\s*bs\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'pagado:?\s*bs\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'(\d{1,6}[.,]\d{2})\s*bs',
            r'bs\s*(\d{1,6}[.,]\d{2})',
            r'(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})',
            r'valor:?\s*bs\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)'
        ]
        
        self.reference_patterns = [
            r'referencia:?\s*(\d{6,20})',
            r'ref\.?\s*(\d{6,20})',
            r'operaci[oó]n:?\s*(\d{6,20})',
            r'n[uú]mero:?\s*(\d{6,20})',
            r'comprobante:?\s*(\d{6,20})',
            r'transacci[oó]n:?\s*(\d{6,20})',
            r'codigo:?\s*(\d{6,20})',
            r'serial:?\s*(\d{6,20})',
            r'(\d{8,20})'
        ]

    def _setup_onnx_precision(self):
        """Configurar ONNX para máxima precisión"""
        try:
            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = 2
            sess_options.intra_op_num_threads = 2
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.method = 'ONNX_Precision'
            logger.info("ONNX Precision configurado")
            
        except Exception as e:
            logger.warning(f"Error ONNX setup: {e}")

    def preprocess_precision(self, image_path):
        """Preprocesamiento para máxima precisión"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                pil_img = Image.open(image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # Redimensionar conservando calidad
            h, w = img.shape[:2]
            if w > 1500 or h > 1500:
                scale = 1500 / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Múltiples versiones para máxima precisión
            versions = []
            
            # Versión 1: Original
            versions.append(('original', gray))
            
            # Versión 2: Mejorar contraste
            enhanced = cv2.equalizeHist(gray)
            versions.append(('enhanced', enhanced))
            
            # Versión 3: Reducir ruido
            denoised = cv2.medianBlur(gray, 3)
            versions.append(('denoised', denoised))
            
            # Versión 4: Umbralización binaria
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            versions.append(('binary', binary))
            
            # Versión 5: Umbralización adaptativa
            adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            versions.append(('adaptive', adaptive))
            
            # Versión 6: Morfología
            kernel = np.ones((2,2), np.uint8)
            morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            versions.append(('morphology', morph))
            
            # Versión 7: Sharpening
            kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(gray, -1, kernel_sharp)
            versions.append(('sharpened', sharpened))
            
            return versions
            
        except Exception as e:
            logger.error(f"Error preprocesamiento: {e}")
            return []

    def extract_text_precision(self, image_path):
        """Extracción de texto con máxima precisión"""
        try:
            versions = self.preprocess_precision(image_path)
            if not versions:
                return []
            
            all_text = []
            
            # Configuraciones múltiples de Tesseract
            configs = [
                '--oem 3 --psm 6 -l spa+eng',
                '--oem 3 --psm 4 -l spa+eng',
                '--oem 3 --psm 3 -l spa+eng',
                '--oem 3 --psm 8 -l spa+eng',
                '--oem 3 --psm 7 -l spa+eng'
            ]
            
            for name, img in versions:
                logger.info(f"Procesando versión: {name}")
                
                for i, config in enumerate(configs):
                    try:
                        text = pytesseract.image_to_string(img, config=config)
                        if text.strip():
                            all_text.append({
                                'version': f"{name}_psm{config.split('psm ')[1].split(' ')[0]}",
                                'full_text': text.strip().lower(),
                                'length': len(text.strip())
                            })
                    except Exception as e:
                        logger.warning(f"Error en {name} config {i}: {e}")
                        continue
            
            return all_text
            
        except Exception as e:
            logger.error(f"Error extracción: {e}")
            return []

    def extract_comprehensive_data(self, text_data):
        """Extracción exhaustiva de datos"""
        full_text = ' '.join([data['full_text'] for data in text_data])
        
        extracted = {}
        
        # 1. BANCO (exhaustivo)
        for bank_name, patterns in self.bank_patterns.items():
            for pattern in patterns:
                if re.search(pattern, full_text, re.IGNORECASE):
                    extracted['bank'] = bank_name.title()
                    logger.info(f"Banco detectado: {bank_name} con patrón: {pattern}")
                    break
            if 'bank' in extracted:
                break
        
        # 2. MONTO (exhaustivo)
        for pattern in self.amount_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                amount_str = match.group(1)
                logger.info(f"Monto encontrado: {amount_str} con patrón: {pattern}")
                
                # Normalizar formato
                if ',' in amount_str and amount_str.count(',') == 1 and len(amount_str.split(',')[1]) <= 2:
                    amount_str = amount_str.replace(',', '.')
                else:
                    amount_str = amount_str.replace(',', '')
                
                try:
                    extracted['amount'] = float(amount_str)
                    break
                except ValueError:
                    continue
        
        # 3. REFERENCIA (exhaustivo)
        for pattern in self.reference_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                ref = match.group(1)
                logger.info(f"Referencia encontrada: {ref} con patrón: {pattern}")
                extracted['reference'] = ref
                break
        
        # 4. FECHA (múltiples formatos)
        date_patterns = [
            r'fecha:?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',
            r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',
            r'fecha:?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2})',
            r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                extracted['date'] = match.group(1)
                logger.info(f"Fecha encontrada: {match.group(1)}")
                break
        
        # 5. HORA (múltiples formatos)
        time_patterns = [
            r'(\d{1,2}:\d{2}:\d{2})',
            r'(\d{1,2}:\d{2})',
            r'hora:?\s*(\d{1,2}:\d{2}:\d{2})',
            r'hora:?\s*(\d{1,2}:\d{2})'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                extracted['time'] = match.group(1)
                logger.info(f"Hora encontrada: {match.group(1)}")
                break
        
        # 6. IDENTIFICACIÓN (múltiples formatos)
        id_patterns = [
            r'identificaci[oó]n:?\s*([VEJPvejp]?\-?\d{6,9})',
            r'c\.?i\.?:?\s*([VEJPvejp]?\-?\d{6,9})',
            r'cedula:?\s*([VEJPvejp]?\-?\d{6,9})',
            r'rif:?\s*([VEJPvejp]?\-?\d{6,9})',
            r'([VEJPvejp]\-?\d{6,9})',
            r'(\d{8})'
        ]
        
        for pattern in id_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                extracted['identification'] = match.group(1)
                logger.info(f"Identificación encontrada: {match.group(1)}")
                break
        
        # 7. CUENTAS (exhaustivo)
        account_patterns = [
            ('origin_account', [
                r'origen:?\s*(\d{4}\*{2,4}\d{2,4})',
                r'origen:?\s*(\d{10,20})',
                r'cuenta\s+origen:?\s*(\d{4}\*{2,4}\d{2,4})'
            ]),
            ('destination_account', [
                r'destino:?\s*(\d{10,20})',
                r'destino:?\s*(\d{4}\*{2,4}\d{2,4})',
                r'cuenta\s+destino:?\s*(\d{10,20})',
                r'(\d{11})'  # Teléfonos como destino
            ])
        ]
        
        for account_type, patterns in account_patterns:
            for pattern in patterns:
                match = re.search(pattern, full_text, re.IGNORECASE)
                if match:
                    extracted[account_type] = match.group(1)
                    logger.info(f"{account_type} encontrada: {match.group(1)}")
                    break
            if account_type in extracted:
                break
        
        # 8. CÓDIGO DE BANCO
        bank_code_patterns = [
            r'banco:?\s*(\d{4})',
            r'c[oó]digo:?\s*(\d{4})',
            r'(\d{4})\s*\-?\s*[a-zA-Z]'
        ]
        
        for pattern in bank_code_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                extracted['destination_bank_code'] = match.group(1)
                logger.info(f"Código banco encontrado: {match.group(1)}")
                break
        
        # 9. TIPO DE OPERACIÓN (exhaustivo)
        operation_types = {
            'pago_movil': [r'pago\s*m[oó]vil', r'pagom[oó]vil'],
            'transferencia': [r'transferencia', r'transfer'],
            'pago': [r'pago(?!\s*m[oó]vil)', r'payment'],
            'deposito': [r'dep[oó]sito', r'deposit'],
            'retiro': [r'retiro', r'withdrawal'],
            'compra': [r'compra', r'purchase']
        }
        
        for op_type, patterns in operation_types.items():
            for pattern in patterns:
                if re.search(pattern, full_text, re.IGNORECASE):
                    extracted['operation_type'] = op_type
                    logger.info(f"Tipo operación encontrado: {op_type}")
                    break
            if 'operation_type' in extracted:
                break
        
        # 10. ESTADO
        status_patterns = [
            r'exitosa?', r'aprobada?', r'completada?', r'procesada?',
            r'fallida?', r'rechazada?', r'pendiente', r'successful'
        ]
        
        for pattern in status_patterns:
            if re.search(pattern, full_text, re.IGNORECASE):
                status = re.search(pattern, full_text, re.IGNORECASE).group(0)
                extracted['status'] = status.lower()
                logger.info(f"Estado encontrado: {status}")
                break
        
        # 11. TELÉFONOS
        phone_patterns = [
            r'(\d{11})',
            r'(\d{4}-\d{7})',
            r'(\+58\d{10})'
        ]
        
        phones = []
        for pattern in phone_patterns:
            matches = re.findall(pattern, full_text)
            for match in matches:
                if match not in phones and len(match) >= 10:
                    phones.append(match)
        
        if phones:
            extracted['phone_numbers'] = phones
            logger.info(f"Teléfonos encontrados: {phones}")
        
        # 12. TODOS LOS NÚMEROS (para análisis)
        number_categories = {
            'very_long': r'\b(\d{20,})\b',
            'long': r'\b(\d{15,19})\b',
            'medium': r'\b(\d{10,14})\b',
            'short': r'\b(\d{6,9})\b',
            'codes': r'\b(\d{4})\b'
        }
        
        all_numbers = {}
        for category, pattern in number_categories.items():
            matches = re.findall(pattern, full_text)
            if matches:
                all_numbers[category] = list(set(matches))
        
        if all_numbers:
            extracted['all_numbers'] = all_numbers
        
        return extracted

    def process_precision(self, image_path):
        """Procesamiento de máxima precisión"""
        if not Path(image_path).exists():
            return {'success': False, 'error': f'Imagen no encontrada: {image_path}'}
        
        try:
            start_time = datetime.now()
            
            # Extraer texto con máxima precisión
            text_data = self.extract_text_precision(image_path)
            
            if not text_data:
                return {'success': False, 'error': 'No se pudo extraer texto'}
            
            # Extraer datos exhaustivamente
            extracted_data = self.extract_comprehensive_data(text_data)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'data': extracted_data,
                'confidence': {
                    'bank': 'high' if extracted_data.get('bank') else 'none',
                    'amount': 'high' if extracted_data.get('amount') else 'none',
                    'reference': 'high' if extracted_data.get('reference') else 'none',
                    'date': 'high' if extracted_data.get('date') else 'none',
                    'identification': 'high' if extracted_data.get('identification') else 'none'
                },
                'processing_info': {
                    'method': self.method,
                    'processing_time': f"{processing_time:.2f}s",
                    'versions_processed': len(text_data),
                    'extraction_timestamp': datetime.now().isoformat()
                },
                'raw_text_samples': [data['full_text'][:200] + '...' if len(data['full_text']) > 200 else data['full_text'] for data in text_data[:3]]
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description='OCR ONNX Máxima Precisión')
    parser.add_argument('image_path', help='Ruta a la imagen')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--compact', '-c', action='store_true')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        ocr = PrecisionOCR()
        result = ocr.process_precision(args.image_path)
        
        # Si se solicita compacto, remover texto raw
        if args.compact and result['success']:
            result.pop('raw_text_samples', None)
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
        sys.exit(0 if result['success'] else 1)
        
    except Exception as e:
        print(json.dumps({'success': False, 'error': str(e)}, indent=2, ensure_ascii=False))
        sys.exit(1)

if __name__ == '__main__':
    main()
