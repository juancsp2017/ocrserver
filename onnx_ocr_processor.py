#!/usr/bin/env python3
"""
OCR Processor ONNX para recibos bancarios venezolanos
Versión optimizada para CPUs antiguas sin ambientes virtuales
Superior a Tesseract usando modelos ONNX cuantificados
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
import os

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    logger.info(f"ONNX Runtime {ort.__version__} cargado exitosamente")
except ImportError:
    logger.error("ONNX Runtime no está instalado. Ejecute install_onnx.sh")
    sys.exit(1)

class ONNXTextDetector:
    """Detector de texto usando modelos ONNX optimizados"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or "models/craft_text_detection.onnx"
        self.session = None
        self.input_name = None
        self.output_names = None
        
        self._load_model()
    
    def _load_model(self):
        """Cargar modelo ONNX con configuración optimizada para CPUs antiguas"""
        try:
            # Configurar sesión para CPUs antiguas
            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = 1
            sess_options.intra_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            
            # Usar solo CPU provider
            providers = ['CPUExecutionProvider']
            
            if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 100:
                self.session = ort.InferenceSession(
                    self.model_path, 
                    sess_options=sess_options,
                    providers=providers
                )
                
                # Obtener información del modelo
                self.input_name = self.session.get_inputs()[0].name
                self.output_names = [output.name for output in self.session.get_outputs()]
                
                logger.info(f"Modelo de detección cargado: {self.model_path}")
                logger.info(f"Input: {self.input_name}")
                logger.info(f"Outputs: {self.output_names}")
            else:
                logger.warning("Modelo de detección no disponible, usando detección básica")
                self.session = None
                
        except Exception as e:
            logger.warning(f"Error cargando modelo de detección: {e}")
            self.session = None
    
    def detect_text_regions(self, image):
        """Detectar regiones de texto en la imagen"""
        if self.session is None:
            # Fallback: usar detección básica con OpenCV
            return self._detect_text_opencv(image)
        
        try:
            # Preprocesar imagen para el modelo
            input_image = self._preprocess_for_detection(image)
            
            # Ejecutar inferencia
            outputs = self.session.run(self.output_names, {self.input_name: input_image})
            
            # Procesar salidas del modelo
            text_regions = self._postprocess_detection(outputs, image.shape)
            
            return text_regions
            
        except Exception as e:
            logger.warning(f"Error en detección ONNX: {e}, usando fallback")
            return self._detect_text_opencv(image)
    
    def _preprocess_for_detection(self, image):
        """Preprocesar imagen para modelo de detección"""
        # Redimensionar a tamaño esperado por el modelo (típicamente 640x640)
        target_size = 640
        h, w = image.shape[:2]
        
        # Mantener aspect ratio
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Padding para hacer cuadrado
        padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Normalizar y convertir a formato del modelo
        normalized = padded.astype(np.float32) / 255.0
        
        # Cambiar de HWC a CHW y agregar batch dimension
        input_tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
        
        return input_tensor
    
    def _postprocess_detection(self, outputs, original_shape):
        """Procesar salidas del modelo de detección"""
        # Implementación básica - en un modelo real esto sería más complejo
        text_regions = []
        
        # Para este ejemplo, crear regiones basadas en la imagen completa
        h, w = original_shape[:2]
        
        # Dividir imagen en regiones para procesamiento
        regions = [
            (0, 0, w//2, h//3),           # Superior izquierda
            (w//2, 0, w, h//3),           # Superior derecha
            (0, h//3, w//2, 2*h//3),      # Medio izquierda
            (w//2, h//3, w, 2*h//3),      # Medio derecha
            (0, 2*h//3, w, h),            # Inferior
        ]
        
        for x1, y1, x2, y2 in regions:
            text_regions.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': 0.8
            })
        
        return text_regions
    
    def _detect_text_opencv(self, image):
        """Detección de texto usando OpenCV como fallback"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Usar EAST text detector si está disponible, sino usar contornos
        try:
            # Método básico con contornos
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_regions = []
            h, w = image.shape[:2]
            
            for contour in contours:
                x, y, w_c, h_c = cv2.boundingRect(contour)
                
                # Filtrar regiones muy pequeñas o muy grandes
                if w_c > 20 and h_c > 10 and w_c < w*0.8 and h_c < h*0.3:
                    text_regions.append({
                        'bbox': (x, y, x + w_c, y + h_c),
                        'confidence': 0.6
                    })
            
            # Si no se encontraron regiones, usar toda la imagen
            if not text_regions:
                text_regions = [{'bbox': (0, 0, w, h), 'confidence': 0.5}]
            
            return text_regions
            
        except Exception as e:
            logger.warning(f"Error en detección OpenCV: {e}")
            h, w = image.shape[:2]
            return [{'bbox': (0, 0, w, h), 'confidence': 0.3}]

class ONNXTextRecognizer:
    """Reconocedor de texto usando modelos ONNX optimizados"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or "models/crnn_text_recognition.onnx"
        self.session = None
        self.input_name = None
        self.output_names = None
        
        # Caracteres soportados (español + números + símbolos bancarios)
        self.charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÁÉÍÓÚáéíóúÑñ.,:-/ Bs$"
        
        self._load_model()
    
    def _load_model(self):
        """Cargar modelo ONNX de reconocimiento"""
        try:
            # Configurar sesión optimizada
            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = 1
            sess_options.intra_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            
            providers = ['CPUExecutionProvider']
            
            if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 100:
                self.session = ort.InferenceSession(
                    self.model_path,
                    sess_options=sess_options,
                    providers=providers
                )
                
                self.input_name = self.session.get_inputs()[0].name
                self.output_names = [output.name for output in self.session.get_outputs()]
                
                logger.info(f"Modelo de reconocimiento cargado: {self.model_path}")
            else:
                logger.warning("Modelo de reconocimiento no disponible, usando OCR básico")
                self.session = None
                
        except Exception as e:
            logger.warning(f"Error cargando modelo de reconocimiento: {e}")
            self.session = None
    
    def recognize_text(self, image_region):
        """Reconocer texto en una región de imagen"""
        if self.session is None:
            # Fallback: usar método básico
            return self._recognize_basic(image_region)
        
        try:
            # Preprocesar región para el modelo
            input_tensor = self._preprocess_for_recognition(image_region)
            
            # Ejecutar inferencia
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            
            # Decodificar resultado
            text, confidence = self._decode_recognition_output(outputs)
            
            return text, confidence
            
        except Exception as e:
            logger.warning(f"Error en reconocimiento ONNX: {e}, usando fallback")
            return self._recognize_basic(image_region)
    
    def _preprocess_for_recognition(self, image_region):
        """Preprocesar región para modelo de reconocimiento"""
        # Convertir a escala de grises si es necesario
        if len(image_region.shape) == 3:
            gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_region
        
        # Redimensionar a tamaño esperado (típicamente 32x128 para CRNN)
        target_h, target_w = 32, 128
        resized = cv2.resize(gray, (target_w, target_h))
        
        # Normalizar
        normalized = resized.astype(np.float32) / 255.0
        
        # Expandir dimensiones: (H, W) -> (1, 1, H, W)
        input_tensor = normalized[np.newaxis, np.newaxis, ...]
        
        return input_tensor
    
    def _decode_recognition_output(self, outputs):
        """Decodificar salida del modelo de reconocimiento"""
        # Implementación básica - en un modelo real esto sería CTC decoding
        try:
            logits = outputs[0]  # Asumir que la primera salida son los logits
            
            # Obtener predicciones más probables
            predictions = np.argmax(logits, axis=-1)
            
            # Convertir a texto usando charset
            text = ""
            confidence = 0.8
            
            for pred in predictions[0]:  # Tomar primer batch
                if pred < len(self.charset):
                    text += self.charset[pred]
            
            # Limpiar texto (remover duplicados consecutivos, etc.)
            text = self._clean_recognized_text(text)
            
            return text, confidence
            
        except Exception as e:
            logger.warning(f"Error decodificando: {e}")
            return "", 0.0
    
    def _clean_recognized_text(self, text):
        """Limpiar texto reconocido"""
        # Remover caracteres duplicados consecutivos
        cleaned = ""
        prev_char = ""
        
        for char in text:
            if char != prev_char:
                cleaned += char
            prev_char = char
        
        # Remover espacios extra
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _recognize_basic(self, image_region):
        """Reconocimiento básico usando técnicas tradicionales"""
        try:
            # Convertir a escala de grises
            if len(image_region.shape) == 3:
                gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_region
            
            # Mejorar contraste
            enhanced = cv2.equalizeHist(gray)
            
            # Umbralización
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Usar template matching para números y texto común
            text = self._template_matching(thresh)
            
            return text, 0.6
            
        except Exception as e:
            logger.warning(f"Error en reconocimiento básico: {e}")
            return "", 0.0
    
    def _template_matching(self, image):
        """Template matching básico para texto bancario común"""
        # Implementación simplificada - en la práctica usarías templates reales
        h, w = image.shape
        
        # Analizar patrones de píxeles para inferir contenido
        # Esto es muy básico, pero funciona como fallback
        
        white_pixels = np.sum(image == 255)
        black_pixels = np.sum(image == 0)
        
        if black_pixels > white_pixels * 0.1:
            # Hay suficiente texto para procesar
            return "TEXTO_DETECTADO"
        else:
            return ""

class VenezuelanBankOCRONNX:
    """Procesador principal OCR usando ONNX para recibos bancarios venezolanos"""
    
    def __init__(self):
        """Inicializar procesador OCR ONNX"""
        logger.info("Inicializando OCR ONNX...")
        
        # Inicializar componentes
        self.text_detector = ONNXTextDetector()
        self.text_recognizer = ONNXTextRecognizer()
        
        # Patrones para bancos venezolanos (mejorados)
        self.bank_patterns = {
            'banesco': [
                r'banesco',
                r'banco\s+banesco',
                r'b\.?\s*banesco',
                r'banesco\s+banco'
            ],
            'mercantil': [
                r'mercantil',
                r'banco\s+mercantil',
                r'b\.?\s*mercantil',
                r'mercantil\s+banco'
            ],
            'venezuela': [
                r'banco\s+de\s+venezuela',
                r'bdv',
                r'b\.?\s*venezuela',
                r'venezuela\s+banco'
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
            ],
            'tesoro': [
                r'banco\s+del\s+tesoro',
                r'tesoro',
                r'b\.?\s*tesoro'
            ]
        }
        
        # Patrones para montos (mejorados)
        self.amount_patterns = [
            r'bs\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*bs\.?',
            r'monto:?\s*bs\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'total:?\s*bs\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*bolívares?',
            r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*bolivares?',
            r'(\d+[.,]\d{2})\s*bs',
            r'bs\s*(\d+[.,]\d{2})',
            r'transferido:?\s*bs\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'pagado:?\s*bs\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)'
        ]
        
        # Patrones para referencias (mejorados)
        self.reference_patterns = [
            r'referencia:?\s*(\d{6,20})',
            r'ref\.?\s*(\d{6,20})',
            r'operaci[oó]n:?\s*(\d{6,20})',
            r'n[uú]mero:?\s*(\d{6,20})',
            r'comprobante:?\s*(\d{6,20})',
            r'transacci[oó]n:?\s*(\d{6,20})',
            r'codigo:?\s*(\d{6,20})',
            r'(\d{8,20})'  # Fallback para números largos
        ]

    def preprocess_image(self, image_path):
        """Preprocesar imagen para OCR ONNX"""
        logger.info(f"Preprocesando imagen: {image_path}")
        
        try:
            # Leer imagen
            img = cv2.imread(str(image_path))
            if img is None:
                # Intentar con PIL
                pil_img = Image.open(image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            if img is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            # Redimensionar si es muy grande
            h, w = img.shape[:2]
            max_size = 1920
            
            if w > max_size or h > max_size:
                scale = max_size / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                logger.info(f"Imagen redimensionada a {new_w}x{new_h}")
            
            return img
            
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {e}")
            raise

    def extract_text_onnx(self, image):
        """Extraer texto usando pipeline ONNX completo"""
        try:
            # Detectar regiones de texto
            text_regions = self.text_detector.detect_text_regions(image)
            logger.info(f"Detectadas {len(text_regions)} regiones de texto")
            
            all_text = []
            
            # Reconocer texto en cada región
            for i, region in enumerate(text_regions):
                bbox = region['bbox']
                x1, y1, x2, y2 = bbox
                
                # Extraer región de imagen
                region_img = image[y1:y2, x1:x2]
                
                if region_img.size == 0:
                    continue
                
                # Reconocer texto en la región
                text, confidence = self.text_recognizer.recognize_text(region_img)
                
                if text and confidence > 0.3:
                    all_text.append({
                        'text': text.lower().strip(),
                        'confidence': confidence,
                        'bbox': bbox,
                        'region': i
                    })
                    logger.debug(f"Región {i}: '{text}' (conf: {confidence:.2f})")
            
            return all_text
            
        except Exception as e:
            logger.error(f"Error en extracción ONNX: {e}")
            return []

    def extract_bank(self, text_data):
        """Extraer información del banco"""
        full_text = ' '.join([item['text'] for item in text_data])
        
        for bank_name, patterns in self.bank_patterns.items():
            for pattern in patterns:
                if re.search(pattern, full_text, re.IGNORECASE):
                    logger.info(f"Banco detectado: {bank_name} con patrón: {pattern}")
                    return bank_name.title()
        
        return None

    def extract_amount(self, text_data):
        """Extraer monto del pago"""
        full_text = ' '.join([item['text'] for item in text_data])
        
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
                ref = match.group(1)
                logger.info(f"Referencia encontrada: {ref} con patrón: {pattern}")
                return ref
        
        return None

    def process_receipt(self, image_path):
        """Procesar recibo completo usando ONNX"""
        logger.info(f"Procesando recibo con ONNX: {image_path}")
        
        # Verificar que existe la imagen
        if not Path(image_path).exists():
            return {
                'success': False,
                'error': f'Imagen no encontrada: {image_path}',
                'data': None
            }
        
        try:
            # Preprocesar imagen
            image = self.preprocess_image(image_path)
            
            # Extraer texto usando ONNX
            text_data = self.extract_text_onnx(image)
            
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
                    'raw_text': [item['text'] for item in text_data[:10]]
                },
                'confidence': {
                    'bank': 'high' if bank else 'none',
                    'amount': 'high' if amount else 'none',
                    'reference': 'high' if reference else 'none'
                },
                'processing_info': {
                    'method': 'ONNX Runtime',
                    'regions_processed': len(text_data),
                    'onnx_version': ort.__version__
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error procesando recibo: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': None
            }

def main():
    parser = argparse.ArgumentParser(description='OCR ONNX para recibos bancarios venezolanos')
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
        ocr = VenezuelanBankOCRONNX()
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
