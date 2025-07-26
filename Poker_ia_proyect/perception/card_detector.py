import cv2
import numpy as np
import pytesseract
import logging
import json
import os
import time
from typing import List, Tuple, Optional, Dict

# ===============================
# CONFIGURACIÓN Y CONSTANTES
# ===============================

# Configuración OCR
OCR_VALOR_CONF = '--psm 10 -c tessedit_char_whitelist=AKQJ1098765432'
OCR_PALO_CONF = '--psm 10'

# Valores válidos de cartas
VALORES_VALIDOS = {"A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"}

# Mapeo de símbolos y letras de palos
PALOS_SIMBOLOS = {"♠": "pica", "♣": "trébol", "♥": "corazón", "♦": "diamante"}
PALOS_LETRAS = {"S": "pica", "C": "trébol", "H": "corazón", "D": "diamante"}

# Áreas de recorte para cartas (coordenadas relativas x, y, w, h)
CARD_AREAS = {
    "player_card_1": (0.30, 0.80, 0.06, 0.12),
    "player_card_2": (0.38, 0.80, 0.06, 0.12),
    "flop_1":       (0.43, 0.47, 0.05, 0.10),
    "flop_2":       (0.49, 0.47, 0.05, 0.10),
    "flop_3":       (0.55, 0.47, 0.05, 0.10),
    "turn":         (0.61, 0.47, 0.05, 0.10),
    "river":        (0.67, 0.47, 0.05, 0.10),
}

# Margen adicional para recortes (porcentaje)
MARGEN_ADICIONAL = 0.08

# Resolución mínima requerida
MIN_RESOLUTION = (480, 640)

# Máximo número de fallos consecutivos de captura
MAX_CONSECUTIVE_FAILURES = 10

class CardDetector:
    """
    Detector de cartas para Poker Texas Hold'em Heads Up.
    Integra OCR robusto con calibración visual y preprocesado optimizado.
    
    Features:
    - Detección específica de áreas de cartas (jugador, flop, turn, river)
    - OCR especializado para valores y palos por separado
    - Calibración visual interactiva
    - Fallback y retry automático
    - Logging detallado y guardado de fallos para debugging
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa el detector de cartas.
        
        Args:
            config_path: Ruta opcional al archivo de configuración JSON
        """
        # Configuración logging
        self.logger = logging.getLogger(__name__)
        
        # Cargar configuración
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Configuración con valores por defecto
        self.card_areas = config.get('card_areas', CARD_AREAS)
        self.margen_adicional = config.get('margen_adicional', MARGEN_ADICIONAL)
        self.valores_validos = set(config.get('valores_validos', VALORES_VALIDOS))
        self.palos_simbolos = config.get('palos_simbolos', PALOS_SIMBOLOS)
        self.palos_letras = config.get('palos_letras', PALOS_LETRAS)
        self.ocr_valor_conf = config.get('ocr_valor_conf', OCR_VALOR_CONF)
        self.ocr_palo_conf = config.get('ocr_palo_conf', OCR_PALO_CONF)
        
        # Directorio para guardar fallos
        self.guardar_fallos_ruta = config.get('guardar_fallos_ruta', "ocr_failures")
        os.makedirs(self.guardar_fallos_ruta, exist_ok=True)
        
        # Estadísticas de OCR
        self.ocr_times = []
        
        self.logger.info("CardDetector inicializado correctamente")

    def preprocesar_ocr_img(self, img: np.ndarray, tipo: str) -> np.ndarray:
        """
        Preprocesa imagen para OCR (unificado para valor y palo).
        
        Args:
            img: Imagen de entrada
            tipo: 'valor' o 'palo' para aplicar configuración específica
        """
        # Usar canal verde para mejor contraste
        if len(img.shape) == 3:
            canal_verde = img[:, :, 1]
        else:
            canal_verde = img
            
        if tipo == 'valor':
            # Configuración para valores
            resized = cv2.resize(canal_verde, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            blur = cv2.GaussianBlur(resized, (3, 3), 0)
            img_preprocesada = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 8)
        else:  # palo
            # Configuración para palos
            resized = cv2.resize(canal_verde, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            blur = cv2.medianBlur(resized, 3)
            _, img_preprocesada = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        return img_preprocesada

    def retry_ocr(self, img: np.ndarray, config: str, tipo: str) -> str:
        """
        Reintenta OCR con redimensionado mejorado.
        
        Args:
            img: Imagen a procesar
            config: Configuración de Tesseract
            tipo: 'valor' o 'palo'
        """
        # Segundo intento con mayor resolución
        img_resized = cv2.resize(img, None, fx=2.6, fy=2.6, interpolation=cv2.INTER_LINEAR)
        img_preprocesada = self.preprocesar_ocr_img(img_resized, tipo)
        return pytesseract.image_to_string(img_preprocesada, config=config)

    def recortar_region(self, frame: np.ndarray, x_rel: float, y_rel: float, 
                       w_rel: float, h_rel: float) -> np.ndarray:
        """Recorta una región específica del frame usando coordenadas relativas."""
        h, w = frame.shape[:2]
        mx = int(self.margen_adicional * w_rel * w)
        my = int(self.margen_adicional * h_rel * h)
        x = int(x_rel * w) - mx
        y = int(y_rel * h) - my
        w_box = int(w_rel * w) + 2 * mx
        h_box = int(h_rel * h) + 2 * my
        x = max(x, 0)
        y = max(y, 0)
        return frame[y:y + h_box, x:x + w_box]

    def ocr_valor(self, img_valor: np.ndarray) -> Optional[str]:
        """Extrae el valor de una carta usando OCR."""
        img_preprocesada = self.preprocesar_ocr_img(img_valor, 'valor')
        texto_crudo = pytesseract.image_to_string(img_preprocesada, config=self.ocr_valor_conf).strip().upper().replace(" ", "")
        
        # Tolerancia: corrige errores comunes
        texto_crudo = texto_crudo.replace("0", "10").replace("O", "10").replace("1", "A") if len(texto_crudo) == 1 else texto_crudo
        self.logger.debug(f"OCR valor crudo: '{texto_crudo}'")
        
        if texto_crudo in self.valores_validos:
            return texto_crudo
        if texto_crudo == "" or not any(c in texto_crudo for c in "AKQJ123456789"):
            return None
            
        # Fallback: buscar valor válido en el texto
        for k in self.valores_validos:
            if k in texto_crudo:
                return k
        return None

    def ocr_palo(self, img_palo: np.ndarray) -> Optional[str]:
        """Extrae el palo de una carta usando OCR."""
        img_preprocesada = self.preprocesar_ocr_img(img_palo, 'palo')
        texto_crudo = pytesseract.image_to_string(img_preprocesada, config=self.ocr_palo_conf)
        self.logger.debug(f"OCR palo crudo: '{texto_crudo}'")
        
        # Buscar símbolos de palos
        for simbolo, nombre in self.palos_simbolos.items():
            if simbolo in texto_crudo:
                return nombre
                
        # Buscar letras de palos
        for letra, nombre in self.palos_letras.items():
            if letra in texto_crudo.upper():
                return nombre
        return None

    def extraer_valor_y_palo(self, carta_img: np.ndarray, area_key: str, 
                           intento: int = 1) -> Optional[Tuple[str, str]]:
        """Extrae valor y palo de una imagen de carta."""
        h, w = carta_img.shape[:2]
        
        # Dividir imagen en regiones de valor y palo
        region_valor = carta_img[0:int(h*0.33), 0:int(w*0.48)]
        region_palo = carta_img[int(h*0.33):int(h*0.75), 0:int(w*0.46)]

        valor = self.ocr_valor(region_valor)
        palo = self.ocr_palo(region_palo)

        if valor and palo:
            self.logger.info(f"{area_key} | Detectada carta: {valor} de {palo}")
            return (valor, palo)
        else:
            self.logger.warning(f"{area_key} | OCR fallido (Valor={valor} Palo={palo}) en intento {intento}")
            
            if intento == 1:
                # Retry con OCR mejorado
                valor_retry = pytesseract.image_to_string(
                    self.preprocesar_ocr_img(region_valor, 'valor'), 
                    config=self.ocr_valor_conf
                ).strip().upper() if not valor else valor
                
                palo_retry = pytesseract.image_to_string(
                    self.preprocesar_ocr_img(region_palo, 'palo'), 
                    config=self.ocr_palo_conf
                ).strip() if not palo else palo
                
                if valor_retry in self.valores_validos and any(letra in palo_retry.upper() for letra in self.palos_letras):
                    return (valor_retry, self.palos_letras.get(palo_retry.upper()[0], palo_retry))
                
                return self.extraer_valor_y_palo(carta_img, area_key, intento+1)
            else:
                # Guardar imagen fallida para debugging
                fecha_nom = f"{area_key}_{int(time.time())}.png"
                img_path = os.path.join(self.guardar_fallos_ruta, fecha_nom)
                cv2.imwrite(img_path, carta_img)
                self.logger.error(f"{area_key} | Guardada imagen fallida: {img_path}")
            return None

    def dibujar_areas(self, frame: np.ndarray, detecciones: Optional[Dict[str, str]] = None) -> np.ndarray:
        """
        Función unificada para dibujar áreas y detecciones.
        
        Args:
            frame: Frame de entrada
            detecciones: Diccionario opcional con detecciones {area_key: label}
        """
        out = frame.copy()
        h, w = out.shape[:2]
        
        for area_key, (x_rel, y_rel, w_rel, h_rel) in self.card_areas.items():
            x, y = int(x_rel * w), int(y_rel * h)
            ww, hh = int(w_rel * w), int(h_rel * h)
            
            # Dibujar rectángulo
            color = (0, 255, 0) if detecciones else (0, 255, 255)
            cv2.rectangle(out, (x, y), (x+ww, y+hh), color, 2)
            
            # Dibujar etiqueta
            if detecciones and area_key in detecciones:
                label = detecciones[area_key]
                cv2.putText(out, label, (x, y+hh+18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)
            else:
                cv2.putText(out, area_key, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                
        return out

    def detect_and_recognize(self, image: np.ndarray) -> List[Dict]:
        """
        Detecta y reconoce cartas en una imagen.
        
        Args:
            image: Frame de entrada
            
        Returns:
            Lista de detecciones con formato compatible con VisionSystem
        """
        detections = []
        cartas_detectadas = self.detectar_cartas(image)
        
        for i, (area_key, (x_rel, y_rel, w_rel, h_rel)) in enumerate(self.card_areas.items()):
            h, w = image.shape[:2]
            x, y = int(x_rel * w), int(y_rel * h)
            ww, hh = int(w_rel * w), int(h_rel * h)
            
            carta = cartas_detectadas[i] if i < len(cartas_detectadas) else (None, None)
            valor, palo = carta
            
            if valor and palo:
                detections.append({
                    'box': (x, y, ww, hh),
                    'label': f"{valor} {palo}",
                    'value': valor,
                    'suit': palo,
                    'position': area_key,
                    'text': f"{valor}{palo}"
                })
        
        return detections

    def detectar_cartas(self, frame: np.ndarray) -> List[Tuple[Optional[str], Optional[str]]]:
        """
        Busca cartas en las áreas definidas.
        
        Returns:
            Lista de tuplas (valor, palo) para cada área de carta
        """
        resultados = []
        tiempos_ocr = []
        
        for area_key, (x_rel, y_rel, w_rel, h_rel) in self.card_areas.items():
            t0 = time.perf_counter()
            carta_img = self.recortar_region(frame, x_rel, y_rel, w_rel, h_rel)
            self.logger.debug(f"{area_key} | Recorte OK - Proporciones ({x_rel}, {y_rel}, {w_rel}, {h_rel})")
            
            deteccion = self.extraer_valor_y_palo(carta_img, area_key)
            t1 = time.perf_counter()
            
            tiempo_ocr = (t1 - t0) * 1000  # en ms
            tiempos_ocr.append(tiempo_ocr)
            self.logger.info(f"{area_key} | OCR completado en {tiempo_ocr:.2f}ms")
            
            resultados.append(deteccion if deteccion else (None, None))
        
        # Estadísticas de tiempo promedio
        if tiempos_ocr:
            tiempo_promedio = sum(tiempos_ocr) / len(tiempos_ocr)
            self.ocr_times.extend(tiempos_ocr)
            self.logger.info(f"Tiempo promedio de OCR por carta: {tiempo_promedio:.2f}ms")
            
        return resultados

    def visualize_detection(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Visualiza las detecciones de cartas sobre la imagen."""
        vis_img = image.copy()
        for det in detections:
            x, y, w, h = det['box']
            label = det['label']
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis_img, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return vis_img

    def calibrar_areas(self, frame: np.ndarray) -> None:
        """Muestra calibración visual de las áreas de cartas."""
        # Validar resolución
        h, w = frame.shape[:2]
        if h < MIN_RESOLUTION[0] or w < MIN_RESOLUTION[1]:
            self.logger.warning(f"Resolución baja detectada: {w}x{h}. Recomendado: {MIN_RESOLUTION[1]}x{MIN_RESOLUTION[0]} o superior")
        
        out = self.dibujar_areas(frame)
        cv2.imshow("Calibración recortes", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def mostrar_detecciones(self, frame: np.ndarray, cartas: List[Tuple[Optional[str], Optional[str]]]) -> None:
        """Muestra las detecciones en tiempo real."""
        # Crear diccionario de detecciones
        detecciones = {}
        for i, (area_key, _) in enumerate(self.card_areas.items()):
            if i < len(cartas) and cartas[i][0] and cartas[i][1]:
                detecciones[area_key] = f"{cartas[i][0]} {cartas[i][1]}"
            else:
                detecciones[area_key] = "?? ??"
        
        img = self.dibujar_areas(frame, detecciones)
        cv2.imshow("Detección de cartas", img)

    def capturar_desde_camara(self, indice_camara: int = 0, calibrar: bool = False) -> None:
        """Captura desde cámara y detecta cartas en tiempo real."""
        cap = cv2.VideoCapture(indice_camara)
        if not cap.isOpened():
            self.logger.error("No se pudo abrir la cámara")
            return
        
        frame_count = 0
        consecutive_failures = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                self.logger.warning(f"Frame no capturado (fallo #{consecutive_failures})")
                
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    self.logger.error(f"Demasiados fallos consecutivos ({consecutive_failures}). Cerrando captura.")
                    break
                continue
            
            # Reset contador de fallos
            consecutive_failures = 0
            
            # Validar resolución en el primer frame
            if frame_count == 0:
                h, w = frame.shape[:2]
                if h < MIN_RESOLUTION[0] or w < MIN_RESOLUTION[1]:
                    self.logger.warning(f"Resolución baja: {w}x{h}. Recomendado: {MIN_RESOLUTION[1]}x{MIN_RESOLUTION[0]} o superior")
                
            if calibrar:
                self.calibrar_areas(frame)
                calibrar = False
                
            cartas = self.detectar_cartas(frame)
            frame_count += 1
            self.mostrar_detecciones(frame, cartas)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

    def save_config(self, path: str) -> None:
        """Guarda la configuración actual a un archivo JSON."""
        config = {
            'card_areas': self.card_areas,
            'margen_adicional': self.margen_adicional,
            'valores_validos': list(self.valores_validos),
            'palos_simbolos': self.palos_simbolos,
            'palos_letras': self.palos_letras,
            'ocr_valor_conf': self.ocr_valor_conf,
            'ocr_palo_conf': self.ocr_palo_conf,
            'guardar_fallos_ruta': self.guardar_fallos_ruta
        }
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        self.logger.info(f"Configuración guardada en {path}")

    def get_ocr_stats(self) -> Dict[str, float]:
        """Retorna estadísticas de rendimiento de OCR."""
        if not self.ocr_times:
            return {}
        
        return {
            'tiempo_promedio_ms': sum(self.ocr_times) / len(self.ocr_times),
            'tiempo_min_ms': min(self.ocr_times),
            'tiempo_max_ms': max(self.ocr_times),
            'total_detecciones': len(self.ocr_times)
        }

# ---- EJEMPLO DE INTEGRACIÓN CON VISION SYSTEM ----
def integrate_with_vision_system():
    """
    Ejemplo de cómo integrar CardDetector con VisionSystem.
    """
    card_detector = CardDetector()
    
    # En tu vision.py, en el método get_game_state:
    def get_game_state_with_cards(self, frame):
        game_state = {'cards': [], 'chips': [], 'actions': [], 'pot_amount': None}
        
        # Usar CardDetector para detectar cartas
        card_detections = card_detector.detect_and_recognize(frame)
        game_state['cards'] = card_detections
        
        # ... resto de la lógica de detección (chips, actions, etc.)
        
        return game_state

# ---- TESTS UNITARIOS ----
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    
    # Crear detector
    detector = CardDetector()
    
    # Test con imagen
    # img = cv2.imread("test_poker_table.png")
    # if img is not None:
    #     detections = detector.detect_and_recognize(img)
    #     vis = detector.visualize_detection(img, detections)
    #     cv2.imshow("Test CardDetector", vis)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    
    # Test con cámara (cambiar calibrar=True para calibrar áreas)
    detector.capturar_desde_camara(indice_camara=0, calibrar=False)
    
    # Mostrar estadísticas al final
    stats = detector.get_ocr_stats()
    if stats:
        print("Estadísticas de OCR:", stats)