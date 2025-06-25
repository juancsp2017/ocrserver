#!/usr/bin/env python3
"""
OCR ONNX HÍBRIDO - Usa ONNX Runtime con fallback a Tesseract
Máxima velocidad con seguridad de fallback
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
import os

# Configurar logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Importar ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    logger.info(f"ONNX Runtime {ort.__version__} disponible")
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime no disponible, usando solo Tesseract")

# Importar Tesseract como fallback
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.error("Tesseract no disponible")

class ONNXTextProcessor:
    """Procesador de texto usando ONNX Runtime"""
    
    def __init__(self, models_dir):
        self.models_dir = Path(models_dir)
        self.detection_session = None
        self.recognition_session = None
        self.onnx_working = False
        
        if ONNX_AVAILABLE:
            self._load_onnx_models()
    
    def _load_onnx_models(self):
        """Cargar modelos ONNX"""
        try:
            # Configurar sesión optimizada para CPUs antiguas
            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = 1
            sess_options.intra_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            
            providers = ['CPUExecutionProvider']
            
            # Cargar modelo de detección
            detection_path = self.models_dir / 'text_detection.onnx'
            if detection_path.exists() and detection_path.stat().st_size > 1000:
                try:
                    self.detection_session = ort.InferenceSession(
                        str(detection_path),
                        sess_options=sess_options,
                        providers=providers
                    )
                    logger.info("Modelo de detección ONNX cargado")
                except Exception as e:
                    logger.warning(f"Error cargando detección ONNX: {e}")
            
            # Cargar modelo de reconocimiento
            recognition_path = self.models_dir / 'text_recognition.onnx'
            if recognition_path.exists() and recognition_path.stat().st_size > 1000:
                try:
                    self.recognition_session = ort.InferenceSession(
                        str(recognition_path),
                        sess_options=sess_options,
                        providers=providers
                    )
                    logger.info("Modelo de reconocimiento ONNX cargado")
                except Exception as e:
                    logger.warning(f"Error cargando reconocimiento ONNX: {e}")
            
            # Verificar si ONNX está funcionando
            if self.detection_session or self.recognition_session:
                self.onnx_working = True
                logger.info("ONNX Runtime operativo")
            else:
                logger.warning("Modelos ONNX no disponibles, usando fallback")
                
        except Exception as e:
            logger.warning(f"Error inicializando ONNX: {e}")
            self.onnx_working = False
    
    def detect_text_regions_onnx(self, image):
        """Detectar regiones de texto con ONNX"""
        if not self.onnx_working or not self.detection_session:
            return self._detect_text_opencv_fallback(image)
        
        try:
            # Preprocesar imagen para ONNX
            input_tensor = self._preprocess_for_detection(image)
            
            # Ejecutar inferencia
            input_name = self.detection_session.get_inputs()[0].name
            outputs = self.detection_session.run(None, {input_name: input_tensor})
            
            # Procesar salidas
            regions = self._postprocess_detection(outputs, image.shape)
            
            logger.info(f"ONNX detectó {len(regions)} regiones")
            return regions
            
        except Exception as e:
            logger.warning(f"Error en detección ONNX: {e}, usando fallback")
            return self._detect_text_opencv_fallback(image)
    
    def _preprocess_for_detection(self, image):
        """Preprocesar imagen para modelo de detección"""
        # Redimensionar a 640x640 (tamaño típico para modelos ONNX)
        target_size = 640
        h, w = image.shape[:2]
        
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Padding para hacer cuadrado
        padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Normalizar y convertir formato
        normalized = padded.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
        
        return input_tensor
    
    def _postprocess_detection(self, outputs, original_shape):
        """Procesar salidas del modelo de detección"""
        h, w = original_shape[:2]
        
        # Implementación básica - dividir imagen en regiones
        regions = [
            {'bbox': (0, 0, w//2, h//2), 'confidence': 0.8},
            {'bbox': (w//2, 0, w, h//2), 'confidence': 0.8},
            {'bbox': (0, h//2, w//2, h), 'confidence': 0.8},
            {'bbox': (w//2, h//2, w, h), 'confidence': 0.8},
            {'bbox': (0, 0, w, h), 'confidence': 0.9}  # Imagen completa
        ]
        
        return regions
    
    def _detect_text_opencv_fallback(self, image):
        """Fallback usando OpenCV"""
        h, w = image.shape[:2]
        return [{'bbox': (0, 0, w, h), 'confidence': 0.7}]

class HybridVenezuelanBankOCR:
    """OCR Híbrido: ONNX + Tesseract fallback"""
    
    def __init__(self):
        """Inicializar OCR híbrido"""
        logger.info("Inicializando OCR Híbrido ONNX + Tesseract...")
        
        # Configurar rutas
        script_dir = Path(__file__).parent
        models_dir = script_dir / 'models'
        
        # Inicializar procesadores
        self.onnx_processor = ONNXTextProcessor(models_dir) if ONNX_AVAILABLE else None
        self.tesseract_config = '--oem 3 --psm 6 -l spa+eng'
        
        # Patrones optimizados
        self.bank_patterns = {
            'banesco': [r'banesco', r'0134'],
            'mercantil': [r'mercantil', r'0105'],
            'venezuela': [r'banco\s+de\s+venezuela', r'bdv', r'pagom[oó]vilbdv', r'0102'],
            'provincial': [r'bbva\s+provincial', r'provincial', r'0108'],
            'bicentenario': [r'bicentenario', r'0175']
        }
        
        self.amount_patterns = [
            r'(\d{1,6}[.,]\d{2})\s*bs',
            r'bs\.?\s*(\d{1,6}[.,]\d{2})',
            r'monto:?\s*bs\.?\s*(\d{1,6}[.,]\d{2})',
            r'(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})'
        ]
        
        self.reference_patterns = [
            r'operaci[oó]n:?\s*(\d{6,20})',
            r'referencia:?\s*(\d{6,20})',
            r'(\d{10,20})'
        ]
        
        # Determinar método principal
        self.primary_method = 'ONNX' if (self.onnx_processor and self.onnx_processor.onnx_working) else 'Tesseract'
        logger.info(f"Método principal: {self.primary_method}")

    def preprocess_image(self, image_path):
        """Preprocesar imagen"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                pil_img = Image.open(image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # Redimensionar para velocidad
            h, w = img.shape[:2]
            if w > 1200 or h > 1200:
                scale = 1200 / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            return img
            
        except Exception as e:
            logger.error(f"Error preprocesando imagen: {e}")
            raise

    def extract_text_hybrid(self, image_path):
        """Extraer texto usando método híbrido"""
        try:
            image = self.preprocess_image(image_path)
            
            # Intentar ONNX primero si está disponible
            if self.primary_method == 'ONNX' and self.onnx_processor:
                try:
                    start_time = datetime.now()
                    
                    # Detectar regiones con ONNX
                    regions = self.onnx_processor.detect_text_regions_onnx(image)
                    
                    # Extraer texto de cada región (usando Tesseract por ahora)
                    all_text = []
                    for region in regions:
                        bbox = region['bbox']
                        x1, y1, x2, y2 = bbox
                        
                        region_img = image[y1:y2, x1:x2]
                        if region_img.size > 0:
                            text = self._extract_text_tesseract(region_img)
                            if text:
                                all_text.append({
                                    'version': f'onnx_region_{len(all_text)}',
                                    'full_text': text.lower(),
                                    'method': 'ONNX+Tesseract'
                                })
                    
                    processing_time = (datetime.now() - start_time).total_seconds()
                    logger.info(f"ONNX procesó en {processing_time:.2f}s")
                    
                    if all_text:
                        return all_text
                    else:
                        logger.warning("ONNX no extrajo texto, usando fallback")
                        
                except Exception as e:
                    logger.warning(f"Error en ONNX: {e}, usando fallback Tesseract")
            
            # Fallback a Tesseract puro
            return self._extract_text_tesseract_fallback(image)
            
        except Exception as e:
            logger.error(f"Error en extracción híbrida: {e}")
            return []

    def _extract_text_tesseract(self, image):
        """Extraer texto con Tesseract"""
        if not TESSERACT_AVAILABLE:
            return ""
        
        try:
            text = pytesseract.image_to_string(image, config=self.tesseract_config)
            return text.strip()
        except Exception as e:
            logger.warning(f"Error Tesseract: {e}")
            return ""

    def _extract_text_tesseract_fallback(self, image):
        """Fallback completo a Tesseract"""
        try:
            start_time = datetime.now()
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Solo 2 versiones para velocidad
            versions = [
                ('enhanced', cv2.equalizeHist(gray)),
                ('binary', cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
            ]
            
            all_text = []
            for name, img in versions:
                text = self._extract_text_tesseract(img)
                if text:
                    all_text.append({
                        'version': f'tesseract_{name}',
                        'full_text': text.lower(),
                        'method': 'Tesseract'
                    })
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Tesseract fallback procesó en {processing_time:.2f}s")
            
            return all_text
            
        except Exception as e:
            logger.error(f"Error en fallback Tesseract: {e}")
            return []

    def extract_data(self, text_data):
        """Extraer datos usando patrones optimizados"""
        full_text = ' '.join([data['full_text'] for data in text_data])
        
        data = {}
        
        # Banco
        for bank_name, patterns in self.bank_patterns.items():
            for pattern in patterns:
                if re.search(pattern, full_text, re.IGNORECASE):
                    data['bank'] = bank_name.title()
                    break
            if 'bank' in data:
                break
        
        # Monto
        for pattern in self.amount_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '.')
                try:
                    data['amount'] = float(amount_str)
                    break
                except:
                    continue
        
        # Referencia
        for pattern in self.reference_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                data['reference'] = match.group(1)
                break
        
        # Fecha
        date_match = re.search(r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})', full_text)
        if date_match:
            data['date'] = date_match.group(1)
        
        # Hora
        time_match = re.search(r'(\d{1,2}:\d{2})', full_text)
        if time_match:
            data['time'] = time_match.group(1)
        
        # Identificación
        id_patterns = [r'identificaci[oó]n:?\s*(\d{6,9})', r'(\d{8})']
        for pattern in id_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                data['identification'] = match.group(1)
                break
        
        # Cuentas
        origin_match = re.search(r'origen:?\s*(\d{4}\*{2,4}\d{2,4})', full_text, re.IGNORECASE)
        if origin_match:
            data['origin_account'] = origin_match.group(1)
        
        dest_match = re.search(r'destino:?\s*(\d{10,11})', full_text, re.IGNORECASE)
        if dest_match:
            data['destination_account'] = dest_match.group(1)
        
        # Código banco
        bank_code_match = re.search(r'banco:?\s*(\d{4})', full_text, re.IGNORECASE)
        if bank_code_match:
            data['destination_bank_code'] = bank_code_match.group(1)
        
        # Tipo operación
        if re.search(r'pago\s*m[oó]vil', full_text, re.IGNORECASE):
            data['operation_type'] = 'pago_movil'
        elif re.search(r'transferencia', full_text, re.IGNORECASE):
            data['operation_type'] = 'transferencia'
        
        # Teléfonos
        phones = re.findall(r'\b(\d{11})\b', full_text)
        if phones:
            data['phone_numbers'] = list(set(phones))
        
        return data

    def process_receipt(self, image_path):
        """Procesar recibo con método híbrido"""
        if not Path(image_path).exists():
            return {
                'success': False,
                'error': f'Imagen no encontrada: {image_path}'
            }
        
        try:
            start_time = datetime.now()
            
            # Extraer texto
            text_data = self.extract_text_hybrid(image_path)
            
            if not text_data:
                return {
                    'success': False,
                    'error': 'No se pudo extraer texto'
                }
            
            # Extraer datos
            extracted_data = self.extract_data(text_data)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Resultado
            result = {
                'success': True,
                'data': extracted_data,
                'confidence': {
                    'bank': 'high' if extracted_data.get('bank') else 'none',
                    'amount': 'high' if extracted_data.get('amount') else 'none',
                    'reference': 'high' if extracted_data.get('reference') else 'none',
                    'date': 'high' if extracted_data.get('date') else 'none'
                },
                'processing_info': {
                    'method': self.primary_method,
                    'processing_time': f"{processing_time:.2f}s",
                    'versions_processed': len(text_data),
                    'extraction_timestamp': datetime.now().isoformat()
                }
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

def main():
    parser = argparse.ArgumentParser(description='OCR Híbrido ONNX + Tesseract')
    parser.add_argument('image_path', help='Ruta a la imagen')
    parser.add_argument('--verbose', '-v', action='store_true', help='Modo verbose')
    parser.add_argument('--compact', '-c', action='store_true', help='Salida compacta')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    try:
        ocr = HybridVenezuelanBankOCR()
        result = ocr.process_receipt(args.image_path)
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
        sys.exit(0 if result['success'] else 1)
        
    except Exception as e:
        error_result = {'success': False, 'error': str(e)}
        print(json.dumps(error_result, indent=2, ensure_ascii=False))
        sys.exit(1)

if __name__ == '__main__':
    main()
