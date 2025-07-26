import cv2
import numpy as np
import json
import os
import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

# =====================
# CONFIGURACIÓN GLOBAL
# =====================

DEFAULT_PATTERN_SIZE = (9, 6)
DEFAULT_NUM_CAPTURES = 20
DEFAULT_CAPTURE_DELAY = 0.7   # segundos
DEFAULT_CAMERA_ID = 0
DEFAULT_RESOLUTION = (1280, 720)
DEFAULT_CAPTURE_DIR = 'captured'
DEFAULT_PARAMS_FILE = 'camera_params.json'

# Umbral para autodiscard de capturas anguladas/distorsionadas
MAX_AVG_ANGLE_DIFF = 30.0    # grados
MIN_PATTERN_SIZE_RATIO = 0.25  # (patrón debe ocupar al menos % del ancho/alto)

# =========================
# SETUP LOGGING AVANZADO
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("calibrate")

# ==============================
# FUNCIONES DE UTILIDAD Y VALIDACIÓN
# ==============================

def calc_angles_between_rows(corners, pattern_size):
    """
    Calcula los ángulos entre filas consecutivas de esquinas detectadas para estimar si el patrón está demasiado inclinado.

    Parámetros
    ----------
    corners : np.ndarray
        Coordenadas de las esquinas detectadas del patrón (shape: (num_corners, 1, 2)).
    pattern_size : tuple
        Tamaño del patrón de ajedrez como (ancho, alto) = (número de columnas, número de filas).

    Returns
    -------
    angles : list of float
        Lista de ángulos (en grados) entre las filas consecutivas del patrón detectado.
    """
    num_rows = pattern_size[1]
    num_cols = pattern_size[0]
    corners = corners.reshape(-1, num_cols, 2)
    angles = []
    for r in range(num_rows - 1):
        v1 = corners[r+1, 0] - corners[r, 0]
        v2 = corners[r+1, -1] - corners[r, -1]
        angle = np.degrees(
            np.arccos(
                np.clip(
                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8),
                    -1.0, 1.0
                )
            )
        )
        angles.append(angle)
    return angles

def pattern_size_in_frame(corners, image_shape):
    """
    Determina el tamaño relativo del patrón de ajedrez dentro del frame de la imagen.

    Parámetros
    ----------
    corners : np.ndarray
        Coordenadas de las esquinas detectadas del patrón.
    image_shape : tuple
        Tamaño de la imagen como (ancho, alto).

    Returns
    -------
    size_x : float
        Proporción del ancho de la imagen ocupada por el patrón.
    size_y : float
        Proporción del alto de la imagen ocupada por el patrón.
    """
    x_coords = corners[:, :, 0]
    y_coords = corners[:, :, 1]
    w, h = image_shape
    span_x = np.ptp(x_coords)
    span_y = np.ptp(y_coords)
    return span_x/w, span_y/h

def is_pattern_valid(corners, image_shape, pattern_size):
    """
    Evalúa automáticamente la validez geométrica de una detección del patrón de ajedrez.
    Descarta patrones que estén muy inclinados o sean demasiado pequeños respecto a la imagen.

    Parámetros
    ----------
    corners : np.ndarray
        Coordenadas refinadas de las esquinas detectadas.
    image_shape : tuple
        Dimensiones de la imagen (ancho, alto).
    pattern_size : tuple
        Tamaño del patrón de ajedrez (ancho, alto).

    Returns
    -------
    valid : bool
        True si la detección es aceptable para calibración, False si debe descartarse.
    reasons : list of str
        Razones por las cuales fue rechazada la captura (si corresponde).

    Notas
    -----
    - Si el patrón está muy inclinado (ángulos entre filas mayores al umbral) o ocupa poco espacio, se rechaza automáticamente.
    """
    angle_diffs = calc_angles_between_rows(corners, pattern_size)
    avg_angle = np.mean(np.abs(angle_diffs))
    size_x, size_y = pattern_size_in_frame(corners, image_shape)
    valid = True
    reasons = []
    if avg_angle > MAX_AVG_ANGLE_DIFF:
        valid = False
        reasons.append(f"Ángulo excesivo ({avg_angle:.1f}°)")
    if size_x < MIN_PATTERN_SIZE_RATIO or size_y < MIN_PATTERN_SIZE_RATIO:
        valid = False
        reasons.append(f"Patrón demasiado pequeño ({size_x:.2f}, {size_y:.2f})")
    return valid, reasons

def save_calibration_images(images, folder):
    """
    Guarda las imágenes de calibración utilizadas en una carpeta especificada.

    Parámetros
    ----------
    images : list of np.ndarray
        Lista de imágenes a guardar.
    folder : str
        Ruta del directorio donde se guardarán las imágenes.

    Returns
    -------
    paths : list of str
        Lista de rutas de los archivos de imagen guardados.
    """
    os.makedirs(folder, exist_ok=True)
    paths = []
    for idx, img in enumerate(images):
        fname = f"calib_{idx+1:02d}.png"
        fpath = os.path.join(folder, fname)
        cv2.imwrite(fpath, img)
        paths.append(fpath)
    return paths

def save_params_json(data, path):
    """
    Guarda los parámetros de calibración en un archivo JSON legible.

    Parámetros
    ----------
    data : dict
        Diccionario con los parámetros de calibración.
    path : str or Path
        Ruta al archivo JSON donde se guardarán los parámetros.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Parámetros guardados correctamente en {path}")

def undistort_sample_image(params, sample_img):
    """
    Aplica la corrección de distorsión a una imagen usando parámetros calibrados.

    Parámetros
    ----------
    params : dict
        Diccionario con 'camera_matrix', 'dist_coeffs', e 'image_shape' proveniente del archivo JSON.
    sample_img : np.ndarray
        Imagen a corregir.

    Returns
    -------
    undistorted : np.ndarray
        Imagen corregida por distorsión de lente.
    """
    cm = np.array(params['camera_matrix'])
    dc = np.array(params['dist_coeffs'])
    img_shape = tuple(params['image_shape'])
    undistorted = cv2.undistort(sample_img, cm, dc)
    return undistorted

def get_camera_indexes(max_devices=10):
    """
    Busca índices de cámaras disponibles en el sistema.

    Parámetros
    ----------
    max_devices : int, opcional
        Número máximo de dispositivos a testear (por default 10).

    Returns
    -------
    index_list : list of int
        Lista de índices de cámaras encontrados como disponibles.
    """
    index_list = []
    for idx in range(max_devices):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            index_list.append(idx)
            cap.release()
    return index_list

# =====================
# CAPTURA DE IMÁGENES
# =====================

def capture_calibration_images(pattern_size, num_captures, delay, camera_id,
                              resolution, show, capture_dir):
    """
    Captura imágenes para la calibración de la cámara, detectando en tiempo real un patrón
    de ajedrez. Requiere que el usuario acepte solo imágenes válidas geométricamente.

    Parámetros
    ----------
    pattern_size : tuple
        Tamaño del patrón de ajedrez (ancho, alto), número de esquinas internas.
    num_captures : int
        Número de capturas válidas necesarias.
    delay : float
        Tiempo mínimo entre capturas válidas (en segundos).
    camera_id : int
        Índice del dispositivo de cámara a utilizar.
    resolution : tuple of int
        Resolución de captura (ancho, alto).
    show : bool
        Si es True, muestra ventana con la vista de la cámara y feedback.
    capture_dir : str
        Carpeta donde guardar imágenes usadas.

    Returns
    -------
    objpoints : list of np.ndarray or None
        Puntos 3D del patrón detectados en el mundo real.
    imgpoints : list of np.ndarray or None
        Puntos 2D del patrón detectados en la imagen.
    image_shape : tuple or None
        Dimensiones (ancho, alto) de la imagen utilizada para calibrar.
    raw_images : list of np.ndarray or None
        Imágenes brutas usadas para calibrar.

    Notas
    -----
    - Si el usuario cancela la calibración antes de suficiente capturas, devuelve None en todos los campos.
    - Maneja la interrupción por teclado limpiamente.
    """
    logger.info(f"Capturando imágenes de calibración usando patrón {pattern_size}...")
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []
    raw_images = []
    image_shape = None

    os.makedirs(capture_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error("No se pudo abrir la cámara. Disponibles: " + str(get_camera_indexes()))
        sys.exit(1)

    # Establece resolución si es soportada
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    capture_count = 0
    last_capture_time = 0

    logger.info("Presiona ESPACIO para capturar, Q para cancelar/calibrar si ya tienes las imágenes necesarias, Ctrl+C para salir.")
    logger.info(f"Las imágenes aceptadas se guardarán en: {capture_dir}")

    try:
        while capture_count < num_captures:
            t1 = time.time()
            ret, frame = cap.read()
            if not ret:
                logger.warning("Fallo al capturar imagen desde la cámara (puede estar desconectada).")
                continue

            disp = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern_size, None)

            # Estadísticas de cámara
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps < 1: fps = 1.0/(time.time() - t1 + 1e-8)

            # Mostrar info en pantalla
            info_line = f"Resolución: {width}x{height} | FPS~{fps:.1f}"
            cv2.putText(disp, info_line, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,220), 2)

            if found:
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                valid, reasons = is_pattern_valid(corners2, (width, height), pattern_size)
                cv2.drawChessboardCorners(disp, pattern_size, corners2, found)
                if valid:
                    cv2.putText(disp, "OK: ESPACIO=Guardar", (20, height-30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50,240,64), 2)
                else:
                    cv2.putText(disp, "Rechazada: " + " | ".join(reasons),
                                (20, height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            else:
                cv2.putText(disp, "Alinea y enfoca el tablero", (20,height-30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

            cv2.putText(disp, f"Capturas válidas: {capture_count}/{num_captures}",
                        (20, height-70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,90), 2)

            if show:
                cv2.imshow("Calibración de cámara", disp)
            k = cv2.waitKey(1) & 0xFF

            if k == ord('q'):
                logger.warning("Calibración cancelada por usuario. Se saldrá del flujo de captura.")
                break

            # Captura solo si:
            # - patrón detectado,
            # - validado geométricamente,
            # - delay correcto
            if k == 32 and found and valid and (time.time() - last_capture_time) > delay:
                objpoints.append(objp.copy())
                imgpoints.append(corners2)
                raw_images.append(frame.copy())
                image_shape = (width, height)
                fname = f"calib_{capture_count+1:02d}.png"
                cv2.imwrite(str(Path(capture_dir)/fname), frame)
                logger.info(f"Captura guardada ({capture_count+1}/{num_captures}) en {fname}")
                capture_count += 1
                last_capture_time = time.time()

        cap.release()
        cv2.destroyAllWindows()

        if len(objpoints) < max(num_captures, 5):  # al menos 5 capturas
            logger.error("No se capturaron suficientes imágenes válidas. Intenta nuevamente.")
            return None, None, None, None

        logger.info(f"Captura de imágenes completa ({len(objpoints)})")
        return objpoints, imgpoints, image_shape, raw_images
    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()
        logger.warning("Interrupción por teclado. Terminando captura de forma segura.")
        if len(objpoints) >= 5:
            logger.info("Número de capturas mínimas alcanzadas, se continúa con la calibración.")
            return objpoints, imgpoints, image_shape, raw_images
        else:
            logger.error("Se canceló sin suficientes capturas para calibrar.")
            return None, None, None, None

# =====================
# CALIBRACIÓN Y OUTPUT
# =====================

def run_calibration(objpoints, imgpoints, image_shape):
    """
    Realiza la calibración de cámara usando los puntos 2D y 3D recolectados.

    Parámetros
    ----------
    objpoints : list of np.ndarray
        Puntos 3D del patrón en coordenadas del mundo real.
    imgpoints : list of np.ndarray
        Puntos 2D del patrón en las imágenes capturadas.
    image_shape : tuple
        Dimensiones (ancho, alto) de las imágenes usadas.

    Returns
    -------
    params : dict or None
        Diccionario con los parámetros calibrados:
        {
            'camera_matrix': ...,
            'dist_coeffs': ...,
            'image_shape': ...,
            'rvecs': ...,
            'tvecs': ...,
        }
        Si la calibración falla, retorna None.
    """
    logger.info("Ejecutando calibración de cámara...")
    try:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_shape, None, None)
        if not ret:
            logger.error("Calibración fallida. Parámetros no guardados.")
            return None
        logger.info("Calibración exitosa.")
        return {
            "camera_matrix": camera_matrix.astype(float).tolist(),
            "dist_coeffs": dist_coeffs.astype(float).flatten().tolist(),
            "image_shape": [int(image_shape[0]), int(image_shape[1])],
            "rvecs": [rvec.flatten().astype(float).tolist() for rvec in rvecs],
            "tvecs": [tvec.flatten().astype(float).tolist() for tvec in tvecs],
        }
    except Exception as e:
        logger.error(f"Excepción inesperada en calibración: {e}")
        return None

def validation_visual_test(params, test_img):
    """
    Muestra una validación visual de la calibración aplicando undistort sobre una imagen capturada.

    Parámetros
    ----------
    params : dict
        Parámetros calibrados cargados desde JSON.
    test_img : np.ndarray
        Imagen a utilizar para mostrar la comparación original vs. corregida.

    Returns
    -------
    None

    Notas
    -----
    - La ventana puede cerrarse presionando 'q'.
    """
    logger.info("Testeando calibración visualmente ('q' para cerrar ventana)...")
    undist = undistort_sample_image(params, test_img)
    concat = np.hstack([
        cv2.resize(test_img, (undist.shape[1], undist.shape[0])),
        undist
    ])
    cv2.putText(concat, "ORIGINAL", (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50,180,250), 2)
    cv2.putText(concat, "UNDISTORTED", (concat.shape[1]//2+30,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (70,250,70), 2)
    cv2.imshow("Validación visual: Izq=Original, Der=Sin Distorsión", concat)
    while True:
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# =====================
# MAIN Y CLI
# =====================

def parse_args():
    """
    Parsea argumentos de línea de comandos para la configuración de calibración.

    Returns
    -------
    args : argparse.Namespace
        Argumentos parseados del usuario.
    """
    parser = argparse.ArgumentParser(
        description="Calibración de cámara para sistema de visión por computadora de póker heads-up")
    parser.add_argument('--pattern-size', type=int, nargs=2, default=DEFAULT_PATTERN_SIZE,
                        help="Tamaño del patrón de ajedrez (cols, rows) -- esquinas internas. Ej: 9 6")
    parser.add_argument('--num-captures', type=int, default=DEFAULT_NUM_CAPTURES,
                        help="Número de capturas válidas requeridas")
    parser.add_argument('--delay', type=float, default=DEFAULT_CAPTURE_DELAY,
                        help="Delay mínimo entre capturas válidas (segundos)")
    parser.add_argument('--camera-id', type=int, default=DEFAULT_CAMERA_ID,
                        help="Índice del dispositivo de cámara (0,1,2...)")
    parser.add_argument('--resolution', type=int, nargs=2, default=DEFAULT_RESOLUTION,
                        help="Resolución de captura (ancho alto)")
    parser.add_argument('--save-dir', type=str, default=DEFAULT_CAPTURE_DIR,
                        help="Directorio donde guardar las imágenes capturadas")
    parser.add_argument('--params-file', type=str, default=DEFAULT_PARAMS_FILE,
                        help="Archivo destino para parámetros calibrados")
    parser.add_argument('--show', action='store_true', default=False,
                        help="Muestra la vista de la cámara durante la captura")
    parser.add_argument('--fast', action='store_true', default=False,
                        help="Reduce la cantidad de capturas mínimas a 8 (modo rápido)")
    args = parser.parse_args()
    return args

def main():
    """
    Punto de entrada principal del script. Orquesta la captura de imágenes,
    calibración y guardado seguro de parámetros.

    Returns
    -------
    None

    Notas
    -----
    - Maneja robustamente errores, interrupciones y respaldo de archivos.
    - Integra validación visual tras la calibración.
    """
    args = parse_args()

    # Ajuste "rápido"
    if args.fast:
        args.num_captures = max(8, args.num_captures)
        logger.info("Modo rápido activado: se requieren solo 8 capturas válidas.")

    # Captura
    objpoints, imgpoints, image_shape, images = capture_calibration_images(
        tuple(args.pattern_size),
        args.num_captures,
        args.delay,
        args.camera_id,
        tuple(args.resolution),
        args.show,
        args.save_dir
    )

    if objpoints is None or imgpoints is None:
        logger.error("No hubo capturas suficientes, calibración abortada.")
        sys.exit(1)

    # Calibración
    params = run_calibration(objpoints, imgpoints, image_shape)
    if params is None:
        logger.error("No se guardó archivo de parámetros por error de calibración.")
        sys.exit(1)

    # Guarda imágenes usadas (ya se guardan en función de captura)
    logger.info("Imágenes de calibración guardadas en: " + str(args.save_dir))

    try:
        out_json_path = Path(args.params_file)
        if out_json_path.exists():
            backup = out_json_path.with_suffix('.bak.json')
            out_json_path.replace(backup)
            logger.warning(f"Archivo {out_json_path} sobreescrito, respaldo en {backup}")
        save_params_json(params, out_json_path)
    except Exception as e:
        logger.error(f"Error al guardar parámetros: {e}")
        sys.exit(1)

    # Test Visual Post-Calibración
    # (usa la última imagen capturada para mostrar corrección de distorsión)
    if images:
        validation_visual_test(params, images[-1])

    logger.info("Proceso de calibración completo y exitoso.")

if __name__ == "__main__":
    main()