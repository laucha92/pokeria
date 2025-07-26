import cv2
import pytesseract
import numpy as np
import logging
import re
import json
import os
from collections import deque
from typing import Dict, Optional, Any

# --- Configuración ---
RANGO_MIN, RANGO_MAX = 0.1, 300
SMOOTH_FRAMES = 5
JSON_REGIONES_PATH = 'regiones_poker.json'
DEBUG_VISUAL = False

OCR_CORRECCIONES = {
    'O': '0', 'o': '0', 'l': '1', 'I': '1', 'L': '1',
    'S': '5', 's': '5', 'B': '8', ',': '.', '|': '1',
    'Q': '0', 'D': '0', 'G': '6', 'Z': '2', ' ': '',
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def cargar_regiones(json_path=JSON_REGIONES_PATH):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No se encontró {json_path}")
    with open(json_path, 'r') as f:
        regiones = json.load(f)
    return regiones

def corregir_ocr(texto):
    return ''.join(OCR_CORRECCIONES.get(c, c) for c in texto)

def _preprocesar_region(region_img):
    escala = 2
    img = cv2.resize(region_img, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoise = cv2.fastNlMeansDenoising(gris, None, h=32)
    eq = cv2.equalizeHist(denoise)
    binaria = cv2.adaptiveThreshold(
        eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 7
    )
    filtrada = cv2.medianBlur(binaria, 3)
    return filtrada

def _extraer_float_de_texto(texto):
    texto = corregir_ocr(texto.upper())
    patron = r"(\d+(?:[.,]\d+)?)"
    busquedas = re.findall(patron, texto)
    for n in busquedas:
        if n:
            n = n.replace(',', '.')
            try:
                return float(n)
            except ValueError:
                continue
    return None

def validar_valor(valor):
    if valor is None:
        return None
    if RANGO_MIN <= valor <= RANGO_MAX:
        return valor
    return None

class Smoother:
    def __init__(self, window=SMOOTH_FRAMES):
        self.window = window
        self.values = deque(maxlen=window)
    def update(self, value):
        if value is not None:
            self.values.append(value)
        return self.get()
    def get(self):
        return float(np.mean(self.values)) if self.values else None

class ChipDetector:
    """
    Módulo desacoplado de visión para detectar stacks, pot y dealer_button.
    Uso principal: detectar_informacion(img: np.ndarray) -> dict
    """
    def __init__(self,
                 regiones_json: str = JSON_REGIONES_PATH,
                 smoothing_frames: int = SMOOTH_FRAMES,
                 visual_debug: bool = DEBUG_VISUAL):
        self.regiones = cargar_regiones(regiones_json)
        self.smoothers = {k: Smoother(window=smoothing_frames) for k in self.regiones}
        self.visual_debug = visual_debug

    def detectar_region(self, img, reg_key):
        x, y, w, h = self.regiones[reg_key]
        return img[y:y+h, x:x+w]

    def mesa_activa(self, img):
        if 'pot' not in self.regiones:
            return False
        pot_img = self.detectar_region(img, 'pot')
        gray = cv2.cvtColor(pot_img, cv2.COLOR_BGR2GRAY)
        brillo = np.mean(gray)
        return brillo > 12

    def detectar_valor(self, img, reg_key) -> Optional[float]:
        region = self.detectar_region(img, reg_key)
        proc = _preprocesar_region(region)
        texto = pytesseract.image_to_string(proc, lang='eng', config='--psm 7')
        texto_corr = corregir_ocr(texto)
        valor = _extraer_float_de_texto(texto_corr)
        valor_valido = validar_valor(valor)
        valor_suavizado = self.smoothers[reg_key].update(valor_valido)
        return valor_suavizado

    def detectar_informacion(self, img: np.ndarray) -> Dict[str, Optional[float]]:
        """
        Devuelve un dict estándar:
        {
            'stack_player': float,
            'stack_opponent': float,
            'pot': float,
            'dealer_button': int (1/0, opcional)
            ...
        }
        """
        if not self.mesa_activa(img):
            logging.info("Mesa no activa.")
            return {k: None for k in self.regiones}

        info = {}
        regiones_debug = []

        for key in self.regiones:
            if key == 'dealer_button':
                region = self.detectar_region(img, key)
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                info['dealer_button'] = 1 if np.mean(gray) > 60 else 0
                regiones_debug.append((region, f'dealer_button: {info["dealer_button"]}'))
            else:
                val = self.detectar_valor(img, key)
                info[key] = val
                regiones_debug.append((self.detectar_region(img, key), f'{key}: {val if val is not None else "-"}'))

        if self.visual_debug:
            self.mostrar_debug(regiones_debug)
        return info

    def mostrar_debug(self, regiones_debug):
        for idx, (region, label) in enumerate(regiones_debug):
            show = cv2.resize(region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            cv2.putText(
                show, label, (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.imshow(f"Región {idx+1}", show)
        cv2.waitKey(1)