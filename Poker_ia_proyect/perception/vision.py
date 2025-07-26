import cv2
import numpy as np
import pytesseract
import logging
import json
import os
import time
import threading
from collections import deque, Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VisionSystem:
    def __init__(self):
        self.roi_config = {}
        self.templates = {}
        self.tesseract_config = r'--oem 3 --psm 6'
        self.templates_dir = "templates"
        self.chip_colors = {}
        self.chip_colors_file = "chip_colors.json"
        self.debug_mode = False
        self.show_all_frames = True
        self.simulated_mode = False
        self.simulated_image_path = None
        self.fps = 0
        self.last_time = time.time()
        self.action_buttons = []
        self.position_buttons = []
        self.stack_rois = []
        self.frame_history = deque(maxlen=100)
        self.detection_history = deque(maxlen=100)
        self.ocr_methods = ["adaptive", "simple", "clahe"]
        self.ocr_method = "adaptive"
        self.capture_thread = None
        self.frame_queue = deque(maxlen=2)
        self.stop_flag = threading.Event()
        self.last_frame = None
        self.last_frame_hash = None
        self.ocr_stats = Counter()
        self.profile_dir = "profiles"
        self.current_profile = "default"
        self.video_writer = None
        self.record_video = False
        self.video_out_path = "session_record.avi"
        self.load_profile(self.current_profile)
        self.load_templates()
        self.load_chip_colors()
        self.load_dynamic_buttons()
        logging.info("Sistema de visión inicializado.")

    def load_profile(self, profile_name):
        os.makedirs(self.profile_dir, exist_ok=True)
        roi_path = os.path.join(self.profile_dir, f"{profile_name}_roi.json")
        if os.path.exists(roi_path):
            with open(roi_path, 'r') as f:
                self.roi_config = json.load(f)
            logging.info(f"Perfil de ROIs '{profile_name}' cargado.")
        else:
            self.roi_config = {}
            logging.warning(f"Perfil de ROIs '{profile_name}' no encontrado, usando vacío.")

    def save_profile(self, profile_name):
        os.makedirs(self.profile_dir, exist_ok=True)
        roi_path = os.path.join(self.profile_dir, f"{profile_name}_roi.json")
        with open(roi_path, 'w') as f:
            json.dump(self.roi_config, f, indent=4)
        logging.info(f"Perfil de ROIs '{profile_name}' guardado.")

    def switch_profile(self, new_profile):
        self.save_profile(self.current_profile)
        self.current_profile = new_profile
        self.load_profile(new_profile)
        self.load_dynamic_buttons()
        logging.info(f"Perfil cambiado a '{new_profile}'.")

    def auto_adjust_rois(self, old_shape, new_shape):
        """Ajusta todas las ROIs proporcionalmente al cambio de resolución."""
        if not old_shape or not new_shape:
            return
        scale_x = new_shape[1] / old_shape[1]
        scale_y = new_shape[0] / old_shape[0]
        for k, (x, y, w, h) in self.roi_config.items():
            self.roi_config[k] = [
                int(x * scale_x), int(y * scale_y),
                int(w * scale_x), int(h * scale_y)
            ]
        logging.info("ROIs autoajustadas a nueva resolución.")

    def load_templates(self):
        self.templates = {}
        if not os.path.exists(self.templates_dir):
            logging.warning(f"Directorio de plantillas no encontrado: {self.templates_dir}")
            return
        for filename in os.listdir(self.templates_dir):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                try:
                    template_name = filename[:-4]
                    template_path = os.path.join(self.templates_dir, filename)
                    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
                    if template is None:
                        raise Exception(f"No se pudo cargar la plantilla: {filename}")
                    self.templates[template_name] = template
                except Exception as e:
                    logging.error(f"Error al cargar la plantilla {filename}: {e}")
        logging.info(f"Se cargaron {len(self.templates)} plantillas desde {self.templates_dir}")

    def load_chip_colors(self):
        try:
            with open(self.chip_colors_file, 'r') as f:
                self.chip_colors = json.load(f)
            logging.info(f"Colores de fichas cargados desde {self.chip_colors_file}")
        except FileNotFoundError:
            logging.warning(f"Archivo de colores de fichas no encontrado: {self.chip_colors_file}")
        except json.JSONDecodeError:
            logging.error(f"Error al decodificar JSON en {self.chip_colors_file}")

    def save_chip_colors(self):
        try:
            with open(self.chip_colors_file, 'w') as f:
                json.dump(self.chip_colors, f, indent=4)
            logging.info(f"Colores de fichas guardados en {self.chip_colors_file}")
        except Exception as e:
            logging.error(f"Error al guardar los colores de fichas: {e}")

    def load_dynamic_buttons(self):
        self.action_buttons = self.roi_config.get("action_buttons", [
            "fold_button", "check_button", "call_button", "raise_button"
        ])
        self.position_buttons = self.roi_config.get("position_buttons", [
            "dealer_button", "sb_button", "bb_button"
        ])
        self.stack_rois = [k for k in self.roi_config if k.startswith("stack")]

    def get_roi(self, name, frame):
        if name in self.roi_config:
            x, y, w, h = self.roi_config[name]
            h_frame, w_frame = frame.shape[:2]
            x, y = max(0, x), max(0, y)
            w, h = min(w, w_frame - x), min(h, h_frame - y)
            if w <= 0 or h <= 0:
                logging.warning(f"ROI '{name}' fuera de límites.")
                return None
            return frame[y:y+h, x:x+w]
        else:
            logging.warning(f"ROI '{name}' no definida.")
            return None

    def preprocess_ocr(self, frame, method):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if method == "adaptive":
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
        elif method == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            thresh = clahe.apply(gray)
        else:  # simple
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        denoised = cv2.medianBlur(thresh, 3)
        return denoised

    def detect_text(self, frame, psm=6, oem=3):
        # Votación OCR con varios métodos
        results = []
        for method in self.ocr_methods:
            preprocessed = self.preprocess_ocr(frame, method)
            tesseract_config = r'--oem {} --psm {}'.format(oem, psm)
            try:
                text = pytesseract.image_to_string(preprocessed, config=tesseract_config).strip()
                if text:
                    results.append(text)
            except Exception as e:
                logging.error(f"OCR error ({method}): {e}")
        if results:
            # Vota por el resultado más frecuente
            best = Counter(results).most_common(1)[0][0]
            self.ocr_stats['success'] += 1
            return best
        else:
            self.ocr_stats['fail'] += 1
            return ""

    def match_template(self, frame, template_name, threshold=0.8):
        if template_name not in self.templates:
            return None
        template = self.templates[template_name]
        h, w = frame.shape[:2]
        th, tw = template.shape[:2]
        scale_x = w / 640
        scale_y = h / 480
        scaled_template = cv2.resize(template, None, fx=scale_x, fy=scale_y)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(scaled_template, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        if len(loc[0]) > 0:
            return (loc[1][0], loc[0][0])
        else:
            return None

    def draw_detections(self, frame, detections, color=(0, 255, 0), thickness=2):
        for detection in detections:
            x, y, w, h = detection['box']
            label = detection.get('label', '')
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            if label:
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    def draw_rois(self, frame):
        for name, (x, y, w, h) in self.roi_config.items():
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv2.putText(frame, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    def detect_cards(self, frame):
        cards = []
        for i in range(1, 6):
            roi_name = f"community_cards_{i}"
            if roi_name in self.roi_config:
                roi = self.get_roi(roi_name, frame)
                if roi is not None:
                    for template_name in self.templates:
                        if template_name.startswith("community_card_"):
                            match = self.match_template(roi, template_name)
                            if match:
                                cards.append({
                                    'type': 'community',
                                    'template': template_name,
                                    'box': self.roi_config[roi_name]
                                })
                    text = self.detect_text(roi)
                    if text:
                        cards.append({
                            'type': 'community',
                            'text': text,
                            'box': self.roi_config[roi_name]
                        })
        for i in range(1, 3):
            roi_name = f"player_cards_{i}"
            if roi_name in self.roi_config:
                roi = self.get_roi(roi_name, frame)
                if roi is not None:
                    for template_name in self.templates:
                        if template_name.startswith("player_card_"):
                            match = self.match_template(roi, template_name)
                            if match:
                                cards.append({
                                    'type': 'player',
                                    'template': template_name,
                                    'box': self.roi_config[roi_name]
                                })
                    text = self.detect_text(roi)
                    if text:
                        cards.append({
                            'type': 'player',
                            'text': text,
                            'box': self.roi_config[roi_name]
                        })
        return cards

    def detect_chips(self, frame):
        chips = []
        stack_rois = self.stack_rois if self.stack_rois else ["pot"]
        for roi_name in stack_rois:
            pot_roi = self.get_roi(roi_name, frame)
            if pot_roi is not None and self.chip_colors:
                hsv = cv2.cvtColor(pot_roi, cv2.COLOR_BGR2HSV)
                for chip_type, color_range in self.chip_colors.items():
                    lower = np.array(color_range['lower'])
                    upper = np.array(color_range['upper'])
                    mask = cv2.inRange(hsv, lower, upper)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in contours:
                        area = cv2.contourArea(c)
                        if area > 20:
                            x, y, w, h = cv2.boundingRect(c)
                            chips.append({'color': chip_type, 'area': area, 'box': (x, y, w, h), 'stack': roi_name})
        return chips

    def detect_actions(self, frame):
        actions = []
        for action_name in self.action_buttons:
            roi = self.get_roi(action_name, frame)
            if roi is not None:
                if action_name in self.templates:
                    match = self.match_template(roi, action_name)
                    if match:
                        actions.append({'action': action_name.replace("_button", ""), 'text': '', 'box': self.roi_config[action_name]})
                        continue
                text = self.detect_text(roi)
                if text:
                    text_lower = text.lower()
                    if "fold" in text_lower:
                        action = "fold"
                    elif "check" in text_lower:
                        action = "check"
                    elif "call" in text_lower:
                        action = "call"
                    elif "raise" in text_lower:
                        action = "raise"
                    else:
                        action = "unknown"
                    actions.append({'action': action, 'text': text, 'box': self.roi_config[action_name]})
        return actions

    def detect_positions(self, frame):
        positions = {}
        for pos in self.position_buttons:
            if pos in self.templates:
                match = self.match_template(frame, pos)
                if match:
                    positions[pos.replace("_button", "")] = match
        return positions

    def detect_pot_amount(self, frame):
        pot_amount = None
        pot_roi = self.get_roi("pot_amount", frame)
        if pot_roi is not None:
            pot_amount = self.detect_text(pot_roi)
            pot_amount = ''.join(filter(str.isdigit, pot_amount))
            if pot_amount:
                self.ocr_stats['pot_success'] += 1
            else:
                self.ocr_stats['pot_fail'] += 1
                pot_amount = None
        return pot_amount

    def get_game_state(self, frame):
        return {
            'cards': self.detect_cards(frame),
            'chips': self.detect_chips(frame),
            'actions': self.detect_actions(frame),
            'positions': self.detect_positions(frame),
            'pot_amount': self.detect_pot_amount(frame)
        }

    def capture_frames(self, cap):
        while not self.stop_flag.is_set():
            ret, frame = cap.read()
            if ret:
                self.frame_queue.append(frame)
            else:
                logging.error("Error al leer el frame.")
                break

    def run(self, camera_index=0):
        if self.simulated_mode:
            if not self.simulated_image_path or not os.path.exists(self.simulated_image_path):
                logging.error("Modo simulado activado, pero no se especificó la ruta de la imagen.")
                return
        else:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                logging.error("No se pudo abrir la cámara.")
                return
            self.capture_thread = threading.Thread(target=self.capture_frames, args=(cap,))
            self.capture_thread.start()
        try:
            prev_frame_hash = None
            prev_shape = None
            while True:
                if self.simulated_mode:
                    frame = cv2.imread(self.simulated_image_path)
                    if frame is None:
                        logging.error(f"No se pudo leer la imagen simulada: {self.simulated_image_path}")
                        break
                else:
                    if not self.frame_queue:
                        time.sleep(0.01)
                        continue
                    frame = self.frame_queue.popleft()
                # Autoajuste de ROIs si cambia la resolución
                if prev_shape and frame.shape != prev_shape:
                    self.auto_adjust_rois(prev_shape, frame.shape)
                prev_shape = frame.shape

                # Procesamiento solo si hay cambios significativos
                frame_hash = hash(frame.tobytes())
                if prev_frame_hash is not None and frame_hash == prev_frame_hash:
                    continue  # No hay cambio, salta procesamiento
                prev_frame_hash = frame_hash

                current_time = time.time()
                self.fps = 1 / (current_time - self.last_time)
                self.last_time = current_time

                vis_frame = frame.copy()
                self.draw_rois(vis_frame)
                game_state = self.get_game_state(frame)
                detections = []
                for card in game_state['cards']:
                    label = f"Card: {card.get('text', card.get('template', ''))}"
                    detections.append({'box': card['box'], 'label': label})
                for chip in game_state['chips']:
                    detections.append({'box': chip['box'], 'label': f"Chip: {chip['color']} ({chip.get('stack', '')})"})
                for action in game_state['actions']:
                    detections.append({'box': action['box'], 'label': f"Action: {action['action']}"})
                annotated_frame = vis_frame.copy()
                if self.show_all_frames:
                    self.draw_detections(annotated_frame, detections)
                else:
                    blank_frame = np.zeros_like(frame)
                    self.draw_detections(blank_frame, detections)
                    annotated_frame = blank_frame
                cv2.putText(annotated_frame, f"FPS: {self.fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"OCR: {'OK' if game_state['pot_amount'] else 'Error'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Poker Vision", annotated_frame)

                # Grabación de video opcional
                if self.record_video:
                    if self.video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        h, w = frame.shape[:2]
                        self.video_writer = cv2.VideoWriter(self.video_out_path, fourcc, 20.0, (w, h))
                    self.video_writer.write(annotated_frame)
                elif self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None

                self.frame_history.append(frame.copy())
                self.detection_history.append({'detections': detections, 'game_state': game_state, 'timestamp': current_time})

                if self.debug_mode and detections:
                    cv2.imwrite(f"debug_frame_{time.time()}.png", annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.define_rois_interactive(frame)
                elif key == ord('c'):
                    chip_type = input("Ingrese el tipo de ficha a calibrar: ")
                    self.calibrate_chip_color(frame, chip_type)
                elif key == ord('s'):
                    self.show_all_frames = not self.show_all_frames
                    logging.info(f"Modo de visualización cambiado a: {'Mostrar todo' if self.show_all_frames else 'Mostrar solo detecciones'}")
                elif key == ord('o'):
                    idx = (self.ocr_methods.index(self.ocr_method) + 1) % len(self.ocr_methods)
                    self.ocr_method = self.ocr_methods[idx]
                    logging.info(f"Método OCR cambiado a: {self.ocr_method}")
                elif key == ord('h'):
                    # Hot-reload de configuración
                    self.load_profile(self.current_profile)
                    self.load_dynamic_buttons()
                    logging.info("Configuración recargada (hot-reload).")
                elif key == ord('p'):
                    # Cambiar perfil
                    new_profile = input("Nombre del nuevo perfil: ")
                    self.switch_profile(new_profile)
                elif key == ord('v'):
                    self.record_video = not self.record_video
                    logging.info(f"Grabación de video {'activada' if self.record_video else 'desactivada'}.")
                elif key == ord('x'):
                    # Mostrar estadísticas OCR
                    print("Estadísticas OCR:", dict(self.ocr_stats))
        except Exception as e:
            logging.error(f"Error en el bucle principal: {e}")
        finally:
            self.stop_flag.set()
            if not self.simulated_mode and self.capture_thread is not None:
                self.capture_thread.join()
                cap.release()
            if self.video_writer is not None:
                self.video_writer.release()
            cv2.destroyAllWindows()
            logging.info("Programa finalizado.")

if __name__ == "__main__":
    vision_system = VisionSystem()
    # vision_system.simulated_mode = True
    # vision_system.simulated_image_path = "test_image.png"
    vision_system.run()