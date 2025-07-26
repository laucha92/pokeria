import logging
import cv2
import copy
import torch
import numpy as np

from config import settings
from game.game_state import GameState
from game.player import Player
from ai.policy import Policy
from ai.neural_model import PokerNet
from perception.vision import detect_cards, detect_chips
from perception.card_detector import CardDetector
from perception.chip_detector import ChipDetector
from utils import data_utils

def initialize_game(config):
    """Inicializa el juego, los jugadores y el estado del juego."""
    player1 = Player(config["player1_name"], config["initial_chips"])
    player2 = Player(config["player2_name"], config["initial_chips"])
    small_blind = config.get("small_blind", 10)
    big_blind = config.get("big_blind", 20)
    game_state = GameState(player1, player2, small_blind, big_blind)
    return player1, player2, game_state

def load_ai_model(config, device):
    """Carga el modelo de IA."""
    model_path = config["ai_model_path"]
    input_size = config["ai_input_size"]
    hidden_size = config["ai_hidden_size"]
    output_size = config["ai_output_size"]
    model = PokerNet(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    policy = Policy(model)
    return policy

def initialize_vision(config):
    """Inicializa los componentes de visión artificial."""
    use_vision = config.get("use_vision", False)
    card_detector = None
    chip_detector = None
    camera_matrix = None
    distortion_coeffs = None

    if use_vision:
        try:
            camera_params_path = config["camera_params_path"]
            camera_params = data_utils.load_data(camera_params_path)
            camera_matrix = camera_params["camera_matrix"]
            distortion_coeffs = camera_params["distortion_coefficients"]
            card_detection_model_path = config["card_detection_model_path"]
            chip_detection_model_path = config["chip_detection_model_path"]
            card_detector = CardDetector(card_detection_model_path, camera_matrix, distortion_coeffs)
            chip_detector = ChipDetector(chip_detection_model_path)
        except Exception as e:
            logging.error(f"Error al inicializar la visión artificial: {e}")
            use_vision = False
            logging.info("Visión artificial desactivada debido a un error.")
    return use_vision, card_detector, chip_detector, camera_matrix, distortion_coeffs

def initialize_camera(config):
    """Inicializa la cámara."""
    use_vision = config.get("use_vision", False)
    cap = None
    if use_vision:
        try:
            camera_id = config.get("camera_id", 0)
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                logging.warning(f"No se pudo abrir la cámara con ID {camera_id}. Reintentando...")
                cap.release()
                cap = None
                cap = cv2.VideoCapture(camera_id)  # Reintentar
                if not cap.isOpened():
                    raise ValueError(f"No se pudo abrir la cámara con ID {camera_id} después de reintentar.")
            logging.info(f"Cámara inicializada con ID {camera_id}.")
        except Exception as e:
            logging.error(f"Error al inicializar la cámara: {e}")
            use_vision = False
            logging.info("Visión artificial desactivada debido a un error en la cámara.")
    return use_vision, cap

def clone_game_state(game_state):
    """
    Crea una copia del estado del juego.
    """
    return copy.deepcopy(game_state)

def get_game_state_key(game_state):
    """
    Devuelve una clave única para el estado del juego.
    """
    # Implementar lógica para generar una clave única basada en el estado del juego
    # Esto podría incluir información como las cartas comunitarias, las apuestas, etc.
    return hash((tuple(game_state.community_cards), game_state.pot, game_state.current_bet, game_state.current_player))