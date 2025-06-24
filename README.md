# OCR para Recibos Bancarios Venezolanos

Solución OCR optimizada para CPUs antiguas usando Tesseract en lugar de EasyOCR/PyTorch.

## 🚀 Instalación Rápida

### 1. Desinstalar versión anterior (si existe)
\`\`\`bash
./uninstall.sh
\`\`\`

### 2. Instalar nueva versión
\`\`\`bash
chmod +x install.sh
./install.sh
\`\`\`

### 3. Verificar instalación
\`\`\`bash
./diagnose.sh
python3 test_ocr.py
\`\`\`

## 💻 Uso

### Básico
\`\`\`bash
./run_ocr.sh /ruta/a/imagen.png
\`\`\`

### Con información detallada
\`\`\`bash
./run_ocr.sh /ruta/a/imagen.png --verbose
\`\`\`

### Para n8n
\`\`\`bash
/home/usuario/venezuelan-bank-ocr/run_ocr.sh "/ruta/completa/a/imagen.png"
\`\`\`

## 🔧 Características

- ✅ **Compatible con CPUs antiguas** (sin AVX/AVX2)
- ✅ **Usa Tesseract** en lugar de PyTorch
- ✅ **Múltiples versiones de procesamiento** de imagen
- ✅ **Patrones específicos** para bancos venezolanos
- ✅ **Salida JSON estructurada**
- ✅ **Optimizado para recursos limitados**

## 📊 Salida JSON

\`\`\`json
{
  "success": true,
  "data": {
    "bank": "Mercantil",
    "amount": 2500.50,
    "reference": "123456789012",
    "raw_text": ["texto extraído..."]
  },
  "confidence": {
    "bank": "high",
    "amount": "high", 
    "reference": "high"
  }
}
\`\`\`

## 🐛 Solución de Problemas

### Error: "Illegal instruction"
Esta versión está diseñada específicamente para evitar este error usando Tesseract.

### Error: "Imagen no encontrada"
Verificar que la ruta sea correcta y que el archivo exista:
\`\`\`bash
ls -la "/ruta/completa/a/imagen.png"
\`\`\`

### Baja precisión
1. Usar `--verbose` para ver detalles
2. Verificar calidad de la imagen
3. Probar con imagen más clara

## 📈 Optimizaciones

- **Sin PyTorch**: Evita problemas de compatibilidad con CPUs antiguas
- **Múltiples procesamientos**: 5 versiones diferentes de cada imagen
- **Patrones flexibles**: Reconoce variaciones en texto bancario
- **Memoria optimizada**: Variables de entorno para limitar threads

## 🔍 Diagnóstico

\`\`\`bash
./diagnose.sh
\`\`\`

Muestra información del sistema, versiones instaladas y crea imagen de prueba.
