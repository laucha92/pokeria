import logging
import os
import sys
import time
import random
from typing import Tuple

import cv2
import torch
import numpy as np

from game.game_state import GameState
from game.player import Player
from ai.policy import Policy, select_action_ai, select_action_human
from ai.neural_model import PokerNet
from perception.card_detector import CardDetector
from perception.chip_detector import ChipDetector
from interface import display, input
from config.settings import handle_config
from utils import logger, data_utils, save_game_history
from exceptions import config_exceptions, game_exceptions, ai_exceptions, vision_exceptions

# Constants
ACTION_BET = "bet"
ACTION_CALL = "call"
ACTION_CHECK = "check"
ACTION_FOLD = "fold"
ACTION_ALL_IN = "all_in"

def run_betting_round(game_state: GameState, players: Tuple[Player, Player], policy: Policy, device: torch.device, round_number: int, game_history: list) -> None:
    """Ejecuta una ronda de apuestas."""
    player1, player2 = players
    player_turn = 1 if round_number % 2 != 0 else 2  # Alternar quién empieza primero
    round_id = f"Round_{round_number}"  # Identificador único por ronda

    while True:
        current_player = player1 if player_turn == 1 else player2
        logging.info(f"Turno de {current_player.name} ({player_turn}) - {round_id}")

        # Obtener la acción del jugador (IA o humano)
        if current_player == player1:  # IA juega como jugador 1
            action, ai_decision_log = select_action_ai(game_state, current_player, policy, device, round_id)
            logging.info(f"IA ({current_player.name}) elige acción: {action} - {round_id}")
        else:
            if config["player2_is_ai"]:
                action, ai_decision_log = select_action_ai(game_state, current_player, policy, device, round_id)
                logging.info(f"IA ({current_player.name}) elige acción: {action} - {round_id}")
            else:
                action = None
                while action is None:
                    action = select_action_human(game_state, current_player)
                ai_decision_log = None  # No hay log de IA para acciones humanas
                logging.info(f"Usuario ({current_player.name}) elige acción: {action} - {round_id}")

        # Registrar la acción específica por ronda
        action_data = {
            "round_id": round_id,
            "player": current_player.name,
            "action": action,
            "amount": game_state.current_bet if action == ACTION_BET else None,
            "order": player_turn
        }
        for round_data in game_history:
            if round_data["round_number"] == round_number:
                round_data["actions"].append(action_data)
                break

        # Aplicar la acción al estado del juego
        try:
            if action == ACTION_BET:
                amount = input.get_bet_amount(game_state.min_bet, current_player.chips)
                game_state.apply_action(current_player, action, amount)
            elif action == ACTION_ALL_IN:
                game_state.apply_action(current_player, action)
            elif action == ACTION_CALL:
                game_state.apply_action(current_player, action)
            elif action == ACTION_CHECK:
                game_state.apply_action(current_player, action)
            elif action == ACTION_FOLD:
                game_state.apply_action(current_player, action)
            else:
                raise ValueError(f"Acción desconocida: {action}")
            logging.info(f"Acción aplicada: {action} - {round_id}")
        except ValueError as e:
            logging.error(f"Error al aplicar la acción: {e} - {round_id}")
            display.display_recommendation(f"Error: {e}")
            continue  # Volver al inicio del bucle

        # Cambiar el turno al siguiente jugador
        player_turn = 2 if player_turn == 1 else 1

        # Verificar si la ronda de apuestas ha terminado
        if game_state.is_betting_round_over():
            break

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
    model_path = os.path.abspath(config["ai_model_path"])
    input_size = config["ai_input_size"]
    hidden_size = config["ai_hidden_size"]
    output_size = config["ai_output_size"]
    model = PokerNet(input_size, hidden_size, output_size)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    policy = Policy(model, device=device)
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
            camera_params_path = os.path.abspath(config["camera_params_path"])
            camera_params = data_utils.load_data(camera_params_path)
            camera_matrix = camera_params["camera_matrix"]
            distortion_coeffs = camera_params["distortion_coefficients"]
            card_detection_model_path = os.path.abspath(config["card_detection_model_path"])
            chip_detection_model_path = os.path.abspath(config["chip_detection_model_path"])
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

def main():
    """
    Función principal que ejecuta el juego de póker con IA y visión artificial.
    """
    config = handle_config()
    start_time = time.time()  # Medir el tiempo de ejecución
    game_history = []  # Inicializar el historial del juego
    player1_wins = 0
    player2_wins = 0
    log = None  # Inicializar log fuera del bloque try
    cap = None  # Inicializar cap fuera del bloque try

    try:
      
        # 2. Inicialización avanzada de logging
        log_level = config.get("log_level", "INFO").upper()  # Obtener el nivel de log desde la configuración
        if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            log_level = "INFO"  # Nivel de log por defecto si no es válido
        log_file = os.path.abspath(config["log_file"])  # Obtener la ruta absoluta
        log = logger.setup_logger("poker_ai", log_file, level=getattr(logging, log_level))  # Logger centralizado

        # Log dual (archivo + consola)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)  # Mostrar warnings y errores en la consola
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        log.addHandler(console_handler)

        log.info("Iniciando juego de póker...")

        # 3. Soporte completo para reproducibilidad
        seed = config.get("seed")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)  # Semilla para numpy
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # Semilla para todas las GPUs
            torch.backends.cudnn.deterministic = True  # Modo determinista para cudnn
            torch.backends.cudnn.benchmark = False  # Desactivar la búsqueda de algoritmos optimizados
            log.info(f"Semilla establecida: {seed}")

        # 4. Inicializar los jugadores y el estado del juego
        player1, player2, game_state = initialize_game(config)
        log.info("Estado del juego inicializado.")

        # 5. Cargar el modelo de IA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = load_ai_model(config, device)
        log.info(f"Modelo de IA cargado en {device}.")

        # 6. Inicializar los componentes de visión artificial
        use_vision, card_detector, chip_detector, camera_matrix, distortion_coeffs = initialize_vision(config)

        # 7. Inicialización condicional de la cámara
        use_vision, cap = initialize_camera(config)

        # 8. Bucle principal del juego
        max_rounds = config.get("max_rounds", 1000)  # Límite de rondas configurables
        round_number = 0
        while not game_state.is_terminal() and round_number < max_rounds:
            round_number += 1
            round_id = f"Round_{round_number}"  # Identificador único por ronda
            log.info(f"--- Ronda {round_number} - {round_id} ---")
            game_state.start_new_round()

            round_data = {
                "round_number": round_number,
                "round_id": round_id,
                "player1_cards": [],
                "player2_cards": [],
                "community_cards": [],
                "actions": [],
                "winner": None
            }

            # 9. Repartir las cartas a los jugadores
            game_state.deal_hole_cards()
            round_data["player1_cards"] = [str(card) for card in player1.hole_cards]
            round_data["player2_cards"] = [str(card) for card in player2.hole_cards]
            log.debug(f"Cartas repartidas: {player1.name} - {player1.hole_cards}, {player2.name} - {player2.hole_cards} - {round_id}")

            # 10. Etapas de apuestas (pre-flop, flop, turn, river)
            stages = ["Pre-Flop", "Flop", "Turn", "River"]
            for stage in stages:
                log.info(f"--- {stage} - {round_id} ---")

                # 11. Repartir cartas comunitarias (si es necesario)
                if stage == "Flop":
                    game_state.deal_flop()
                elif stage == "Turn":
                    game_state.deal_turn()
                elif stage == "River":
                    game_state.deal_river()
                round_data["community_cards"] = [str(card) for card in game_state.community_cards]
                log.debug(f"Cartas comunitarias: {game_state.community_cards} - {round_id}")

                # 12. Bucle de apuestas para la etapa actual
                players = (player1, player2)
                run_betting_round(game_state, players, policy, device, round_number, game_history)

            # 13. Determinar el ganador de la ronda
            winner = game_state.determine_winner()
            log.info(f"El ganador de la ronda es: {winner.name} - {round_id}")
            round_data["winner"] = winner.name
            if winner == player1:
                player1_wins += 1
            else:
                player2_wins += 1
            display.display_game_state(game_state, player_view=None)
            display.display_recommendation(f"El ganador de la ronda es: {winner.name}")

            # Guardar historial de acciones
            game_history.append(round_data)

            # Mostrar estadísticas intermedias
            if round_number % config.get("stats_interval", 10) == 0:
                total_rounds = len(game_history)
                player1_win_rate = (player1_wins / total_rounds) * 100 if total_rounds > 0 else 0
                player2_win_rate = (player2_wins / total_rounds) * 100 if total_rounds > 0 else 0
                log.info(f"--- Estadísticas intermedias (Ronda {round_number}) ---")
                log.info(f"{player1.name} Win Rate: {player1_win_rate:.2f}%")
                log.info(f"{player2.name} Win Rate: {player2_win_rate:.2f}%")

            # 14. Preparar para la siguiente ronda
            game_state.reset_round()

        # 15. Finalizar el juego y mostrar los resultados
        final_winner = game_state.determine_final_winner()
        log.info(f"El ganador final es: {final_winner.name}")
        display.display_game_state(game_state, player_view=None)
        display.display_recommendation(f"El ganador final es: {final_winner.name}")

        # Resumen estadístico final
        total_rounds = len(game_history)
        player1_win_rate = (player1_wins / total_rounds) * 100 if total_rounds > 0 else 0
        player2_win_rate = (player2_wins / total_rounds) * 100 if total_rounds > 0 else 0
        log.info(f"--- Resumen del juego ---")
        log.info(f"Rondas jugadas: {total_rounds}")
        log.info(f"{player1.name} ganó {player1_wins} rondas ({player1_win_rate:.2f}%)")
        log.info(f"{player2.name} ganó {player2_wins} rondas ({player2_win_rate:.2f}%)")

        # Guardar historial del juego en disco
        save_game_history(game_history, config.get("game_history_filename", "game_history"), config.get("game_history_format", "json"))

    except FileNotFoundError as e:
        if log:
            log.error(f"Error de archivo no encontrado: {e}")
        display.display_recommendation(f"Error: Archivo no encontrado - {e}")
        sys.exit(1)
    except config_exceptions.ConfigLoadError as e:
        if log:
            log.error(f"Error al cargar la configuración: {e}")
        display.display_recommendation(f"Error: Configuración inválida - {e}")
        sys.exit(1)
    except game_exceptions.GameException as e:
        if log:
            log.error(f"Error en el juego: {e}")
        display.display_recommendation(f"Error en el juego: {e}")
        sys.exit(1)
    except ai_exceptions.AIException as e:
        if log:
            log.error(f"Error en la IA: {e}")
        display.display_recommendation(f"Error en la IA: {e}")
        sys.exit(1)
    except vision_exceptions.VisionException as e:
        if log:
            log.error(f"Error en la visión: {e}")
        display.display_recommendation(f"Error en la visión: {e}")
        sys.exit(1)
    except ValueError as e:
        if log:
            log.error(f"Error de valor: {e}")
        display.display_recommendation(f"Error: Valor inválido - {e}")
        sys.exit(1)
    except Exception as e:
        if log:
            log.error(f"Error inesperado: {e}")
        display.display_recommendation(f"Error inesperado: {e}")
        sys.exit(1)

    finally:
        if cap is not None and cap.isOpened():  # Verificar si cap está inicializado
            cap.release()  # Liberar la cámara
        end_time = time.time()
        execution_time = end_time - start_time
        if log:
            log.info(f"Juego terminado. Tiempo de ejecución: {execution_time:.2f} segundos.")

if __name__ == "__main__":
    main()