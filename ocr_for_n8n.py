#!/usr/bin/env python3
"""
OCR Completo para n8n - Extrae TODA la información de recibos bancarios venezolanos
Optimizado para máxima extracción de datos
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
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteVenezuelanBankOCR:
    def __init__(self):
        """Inicializar el procesador OCR completo"""
        logger.info("Inicializando OCR Completo para n8n...")
        
        # Configurar Tesseract para español
        self.tesseract_config = '--oem 3 --psm 6 -l spa+eng'
        
        # Patrones para bancos venezolanos (expandidos)
        self.bank_patterns = {
            'banesco': [
                r'banesco',
                r'banco\s+banesco',
                r'b\.?\s*banesco',
                r'banesco\s+banco',
                r'0134'  # Código de banco
            ],
            'mercantil': [
                r'mercantil',
                r'banco\s+mercantil',
                r'b\.?\s*mercantil',
                r'mercantil\s+banco',
                r'0105'  # Código de banco
            ],
            'venezuela': [
                r'banco\s+de\s+venezuela',
                r'bdv',
                r'b\.?\s*venezuela',
                r'venezuela\s+banco',
                r'pagom[oó]vilbdv',
                r'0102'  # Código de banco
            ],
            'provincial': [
                r'bbva\s+provincial',
                r'provincial',
                r'bbva',
                r'banco\s+provincial',
                r'0108'  # Código de banco
            ],
            'bicentenario': [
                r'bicentenario',
                r'banco\s+bicentenario',
                r'b\.?\s*bicentenario',
                r'0175'  # Código de banco
            ],
            'tesoro': [
                r'banco\s+del\s+tesoro',
                r'tesoro',
                r'b\.?\s*tesoro',
                r'0163'  # Código de banco
            ]
        }
        
        # Patrones para montos (expandidos)
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
            r'pagado:?\s*bs\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'(\d{1,6}[.,]\d{2})\s*bs'  # Para montos como 209,08 bs
        ]
        
        # Patrones para referencias/operaciones (expandidos)
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
        
        # Patrones para fechas
        self.date_patterns = [
            r'fecha:?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',
            r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',
            r'fecha:?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2})',
            r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2})'
        ]
        
        # Patrones para horas
        self.time_patterns = [
            r'(\d{1,2}:\d{2}:\d{2})',
            r'(\d{1,2}:\d{2})',
            r'hora:?\s*(\d{1,2}:\d{2}:\d{2})',
            r'hora:?\s*(\d{1,2}:\d{2})'
        ]
        
        # Patrones para cuentas
        self.account_patterns = [
            r'origen:?\s*(\d{4}\*{4}\d{4})',
            r'destino:?\s*(\d{10,20})',
            r'cuenta:?\s*(\d{10,20})',
            r'(\d{4}\*{2,4}\d{2,4})',  # Cuentas enmascaradas
            r'(\d{10,20})'  # Números de cuenta completos
        ]
        
        # Patrones para identificación
        self.id_patterns = [
            r'identificaci[oó]n:?\s*([VEJPvejp]?\-?\d{6,9})',
            r'c\.?i\.?:?\s*([VEJPvejp]?\-?\d{6,9})',
            r'cedula:?\s*([VEJPvejp]?\-?\d{6,9})',
            r'rif:?\s*([VEJPvejp]?\-?\d{6,9})',
            r'([VEJPvejp]\-?\d{6,9})'
        ]
        
        # Patrones para códigos de banco
        self.bank_code_patterns = [
            r'banco:?\s*(\d{4})',
            r'c[oó]digo:?\s*(\d{4})',
            r'(\d{4})\s*\-?\s*[a-zA-Z]'
        ]

    def preprocess_image(self, image_path):
        """Preprocesar imagen para mejorar OCR"""
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
            
            # Crear múltiples versiones procesadas para máxima extracción
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
            
            # Versión 6: Morfología para limpiar texto
            kernel = np.ones((2,2), np.uint8)
            morph = cv2.morphologyEx(thresh_binary, cv2.MORPH_CLOSE, kernel)
            processed_images.append(('morphology', morph))
            
            return processed_images
            
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {e}")
            return []

    def extract_text_tesseract(self, image, config=None):
        """Extraer texto usando Tesseract con configuraciones múltiples"""
        if config is None:
            config = self.tesseract_config
        
        results = []
        
        # Configuraciones múltiples para máxima extracción
        configs = [
            '--oem 3 --psm 6 -l spa+eng',  # Configuración principal
            '--oem 3 --psm 4 -l spa+eng',  # Una columna de texto
            '--oem 3 --psm 3 -l spa+eng',  # Página completa
            '--oem 3 --psm 8 -l spa+eng',  # Una palabra
            '--oem 3 --psm 7 -l spa+eng',  # Una línea de texto
        ]
        
        for i, cfg in enumerate(configs):
            try:
                # Extraer texto
                text = pytesseract.image_to_string(image, config=cfg)
                
                if text.strip():
                    results.append({
                        'config': f'psm_{cfg.split("psm ")[1].split(" ")[0]}',
                        'text': text.strip().lower(),
                        'length': len(text.strip())
                    })
                    
            except Exception as e:
                logger.warning(f"Error en configuración {i}: {e}")
                continue
        
        return results

    def extract_text(self, image_path):
        """Extraer texto usando múltiples versiones de la imagen y configuraciones"""
        try:
            # Preprocesar imagen
            processed_images = self.preprocess_image(image_path)
            
            if not processed_images:
                return []
            
            all_text_data = []
            
            # Procesar cada versión de la imagen
            for name, img in processed_images:
                logger.info(f"Procesando versión: {name}")
                
                results = self.extract_text_tesseract(img)
                
                for result in results:
                    if result['text']:
                        all_text_data.append({
                            'version': f"{name}_{result['config']}",
                            'full_text': result['text'],
                            'length': result['length']
                        })
            
            return all_text_data
            
        except Exception as e:
            logger.error(f"Error extrayendo texto: {e}")
            return []

    def extract_all_data(self, text_data):
        """Extraer TODA la información posible del texto"""
        # Combinar todo el texto
        full_text = ' '.join([data['full_text'] for data in text_data])
        
        extracted_data = {}
        
        # 1. BANCO
        extracted_data['bank'] = self.extract_bank(text_data)
        
        # 2. MONTO
        extracted_data['amount'] = self.extract_amount(text_data)
        
        # 3. REFERENCIA/OPERACIÓN
        extracted_data['reference'] = self.extract_reference(text_data)
        
        # 4. FECHA
        extracted_data['date'] = self.extract_date(full_text)
        
        # 5. HORA
        extracted_data['time'] = self.extract_time(full_text)
        
        # 6. IDENTIFICACIÓN
        extracted_data['identification'] = self.extract_identification(full_text)
        
        # 7. CUENTA ORIGEN
        extracted_data['origin_account'] = self.extract_origin_account(full_text)
        
        # 8. CUENTA DESTINO
        extracted_data['destination_account'] = self.extract_destination_account(full_text)
        
        # 9. CÓDIGO DE BANCO DESTINO
        extracted_data['destination_bank_code'] = self.extract_bank_code(full_text)
        
        # 10. TIPO DE OPERACIÓN
        extracted_data['operation_type'] = self.extract_operation_type(full_text)
        
        # 11. ESTADO DE LA TRANSACCIÓN
        extracted_data['status'] = self.extract_status(full_text)
        
        # 12. NÚMEROS DE TELÉFONO
        extracted_data['phone_numbers'] = self.extract_phone_numbers(full_text)
        
        # 13. TODOS LOS NÚMEROS ENCONTRADOS
        extracted_data['all_numbers'] = self.extract_all_numbers(full_text)
        
        return extracted_data

    def extract_bank(self, text_data):
        """Extraer información del banco"""
        full_text = ' '.join([data['full_text'] for data in text_data])
        
        for bank_name, patterns in self.bank_patterns.items():
            for pattern in patterns:
                if re.search(pattern, full_text, re.IGNORECASE):
                    logger.info(f"Banco detectado: {bank_name} con patrón: {pattern}")
                    return bank_name.title()
        
        return None

    def extract_amount(self, text_data):
        """Extraer monto del pago"""
        full_text = ' '.join([data['full_text'] for data in text_data])
        
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
        full_text = ' '.join([data['full_text'] for data in text_data])
        
        for pattern in self.reference_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                ref = match.group(1)
                logger.info(f"Referencia encontrada: {ref} con patrón: {pattern}")
                return ref
        
        return None

    def extract_date(self, text):
        """Extraer fecha"""
        for pattern in self.date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                logger.info(f"Fecha encontrada: {date_str}")
                return date_str
        return None

    def extract_time(self, text):
        """Extraer hora"""
        for pattern in self.time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                time_str = match.group(1)
                logger.info(f"Hora encontrada: {time_str}")
                return time_str
        return None

    def extract_identification(self, text):
        """Extraer identificación (cédula/RIF)"""
        for pattern in self.id_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                id_str = match.group(1)
                logger.info(f"Identificación encontrada: {id_str}")
                return id_str
        return None

    def extract_origin_account(self, text):
        """Extraer cuenta origen"""
        patterns = [
            r'origen:?\s*(\d{4}\*{2,4}\d{2,4})',
            r'origen:?\s*(\d{10,20})',
            r'(\d{4}\*{2,4}\d{2,4})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                account = match.group(1)
                logger.info(f"Cuenta origen encontrada: {account}")
                return account
        return None

    def extract_destination_account(self, text):
        """Extraer cuenta destino"""
        patterns = [
            r'destino:?\s*(\d{10,20})',
            r'destino:?\s*(\d{4}\*{2,4}\d{2,4})',
            r'(\d{11})'  # Números de teléfono como destino
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                account = match.group(1)
                logger.info(f"Cuenta destino encontrada: {account}")
                return account
        return None

    def extract_bank_code(self, text):
        """Extraer código de banco"""
        for pattern in self.bank_code_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                code = match.group(1)
                logger.info(f"Código de banco encontrado: {code}")
                return code
        return None

    def extract_operation_type(self, text):
        """Extraer tipo de operación"""
        operation_types = {
            'transferencia': r'transferencia',
            'pago_movil': r'pago\s*m[oó]vil',
            'pago': r'pago(?!\s*m[oó]vil)',
            'deposito': r'dep[oó]sito',
            'retiro': r'retiro',
            'compra': r'compra'
        }
        
        for op_type, pattern in operation_types.items():
            if re.search(pattern, text, re.IGNORECASE):
                logger.info(f"Tipo de operación encontrado: {op_type}")
                return op_type
        return None

    def extract_status(self, text):
        """Extraer estado de la transacción"""
        status_patterns = [
            r'exitosa?',
            r'aprobada?',
            r'completada?',
            r'procesada?',
            r'fallida?',
            r'rechazada?',
            r'pendiente'
        ]
        
        for pattern in status_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                status = re.search(pattern, text, re.IGNORECASE).group(0)
                logger.info(f"Estado encontrado: {status}")
                return status.lower()
        return None

    def extract_phone_numbers(self, text):
        """Extraer números de teléfono"""
        phone_patterns = [
            r'(\d{11})',  # 04125318244
            r'(\d{4}-\d{7})',  # 0412-5318244
            r'(\+58\d{10})'  # +584125318244
        ]
        
        phones = []
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match not in phones and len(match) >= 10:
                    phones.append(match)
                    logger.info(f"Teléfono encontrado: {match}")
        
        return phones if phones else None

    def extract_all_numbers(self, text):
        """Extraer todos los números encontrados"""
        # Encontrar todos los números de diferentes longitudes
        number_patterns = [
            r'\b(\d{20,})\b',  # Números muy largos
            r'\b(\d{15,19})\b',  # Números largos
            r'\b(\d{10,14})\b',  # Números medianos
            r'\b(\d{6,9})\b',   # Números cortos
            r'\b(\d{4})\b'      # Códigos
        ]
        
        all_numbers = {}
        
        for i, pattern in enumerate(number_patterns):
            matches = re.findall(pattern, text)
            category = ['very_long', 'long', 'medium', 'short', 'codes'][i]
            if matches:
                all_numbers[category] = list(set(matches))  # Eliminar duplicados
        
        return all_numbers if all_numbers else None

    def process_receipt(self, image_path):
        """Procesar recibo completo extrayendo TODA la información"""
        logger.info(f"Procesando recibo completo: {image_path}")
        
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
        
        # Extraer TODA la información
        extracted_data = self.extract_all_data(text_data)
        
        # Preparar resultado completo para n8n
        result = {
            'success': True,
            'data': extracted_data,
            'confidence': {
                'bank': 'high' if extracted_data['bank'] else 'none',
                'amount': 'high' if extracted_data['amount'] else 'none',
                'reference': 'high' if extracted_data['reference'] else 'none',
                'date': 'high' if extracted_data['date'] else 'none',
                'identification': 'high' if extracted_data['identification'] else 'none'
            },
            'processing_info': {
                'versions_processed': len(text_data),
                'tesseract_version': str(pytesseract.get_tesseract_version()),
                'extraction_timestamp': datetime.now().isoformat()
            },
            'raw_text_samples': [data['full_text'][:300] + '...' if len(data['full_text']) > 300 else data['full_text'] for data in text_data[:3]],
            'all_extracted_text': [data['full_text'] for data in text_data]  # Para análisis adicional en n8n
        }
        
        return result

def main():
    parser = argparse.ArgumentParser(description='OCR Completo para n8n - Recibos bancarios venezolanos')
    parser.add_argument('image_path', help='Ruta a la imagen del recibo')
    parser.add_argument('--verbose', '-v', action='store_true', help='Modo verbose')
    parser.add_argument('--compact', '-c', action='store_true', help='Salida compacta (sin raw text)')
    
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
        ocr = CompleteVenezuelanBankOCR()
        result = ocr.process_receipt(image_path)
        
        # Si se solicita salida compacta, remover texto raw
        if args.compact and result['success']:
            result.pop('all_extracted_text', None)
            result.pop('raw_text_samples', None)
        
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
