from collections import deque
from typing import List, Dict, Tuple, Optional
from enum import Enum
import json
import yaml  # Asegúrate de instalar PyYAML: pip install pyyaml
import logging
import unittest

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InvalidRoundStateError(Exception):
    """Excepción para estados de ronda inválidos."""
    pass

class Strategy(Enum):
    MODEL = "model"
    MCTS = "mcts"
    CFR = "cfr"

class GameContextAnalyzer:
    """
    Analiza el contexto del juego para ajustar los pesos de las estrategias.
    """
    def __init__(self, config_path: str = None):
        """
        Inicializa el analizador de contexto del juego.

        Args:
            config_path (str, optional): Ruta al archivo de configuración (JSON o YAML).
        """
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self.num_players = self.config.get("num_players", 9)
        self.historical_data = deque(maxlen=self.config.get("historical_data_maxlen", 10))
        self.verbose = self.config.get("verbose", False)
        self.min_stack_size = self.config.get("min_stack_size", 10.0)
        self.max_aggression_threshold = self.config.get("max_aggression_threshold", 0.8)

        logging.info(f"GameContextAnalyzer inicializado con configuración: {self.config}")

    def _load_config(self, config_path: str) -> Dict:
        """
        Carga la configuración desde un archivo JSON o YAML.
        """
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    return json.load(f)
                elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    raise ValueError("Formato de archivo no soportado. Use JSON o YAML.")
        except Exception as e:
            logging.error(f"Error al cargar la configuración desde {config_path}: {e}")
            return self._default_config()

    def _default_config(self) -> Dict:
        """
        Devuelve la configuración por defecto.
        """
        return {
            "num_players": 9,
            "historical_data_maxlen": 10,
            "verbose": False,
            "min_stack_size": 10.0,
            "max_aggression_threshold": 0.8
        }

    def analyze_context(self, game_state: List[float], round_state: Dict) -> Tuple[Dict[Strategy, float], Optional[Dict[str, float]]]:
        """
        Analiza el estado del juego y devuelve un diccionario de pesos ajustados.

        Args:
            game_state (List[float]): Estado del juego.
            round_state (Dict): Estado de la ronda.

        Returns:
            Tuple[Dict[Strategy, float], Optional[Dict[str, float]]]:
                - Diccionario de pesos ajustados para cada estrategia.
                - Diccionario opcional de factores intermedios (si verbose es True).
        """
        try:
            self._validate_round_state(round_state)

            # Lógica de análisis de contexto (ejemplo)
            context_weights = {
                Strategy.MODEL: 0.3,
                Strategy.MCTS: 0.4,
                Strategy.CFR: 0.3
            }

            # Actualizar el historial de datos
            self.historical_data.append(round_state)

            # Calcular factores
            aggression_factor = self._calculate_aggression_factor()
            stack_size_factor = self._calculate_stack_size_factor(round_state["stack_sizes"])
            position_factor = self._calculate_position_factor(round_state["current_player_position"])
            round_phase_factor = self._calculate_round_phase_factor(round_state["round_phase"])

            # Ajustar pesos basados en los factores
            context_weights = self._adjust_weights_for_aggression(context_weights, aggression_factor)
            context_weights = self._adjust_weights_for_stack_size(context_weights, stack_size_factor)
            context_weights = self._adjust_weights_for_position(context_weights, position_factor)
            context_weights = self._adjust_weights_for_round_phase(context_weights, round_phase_factor)

            # Normalización final de pesos
            context_weights = self._normalize_weights(context_weights)

            # Registrar factores si verbose es True
            factors = None
            if self.verbose:
                factors = {
                    "aggression_factor": aggression_factor,
                    "stack_size_factor": stack_size_factor,
                    "position_factor": position_factor,
                    "round_phase_factor": round_phase_factor
                }
                logging.info(f"Factores: {factors}")

            return context_weights, factors

        except InvalidRoundStateError as e:
            logging.error(f"Error en analyze_context: {e}")
            return {Strategy.MODEL: 1/3, Strategy.MCTS: 1/3, Strategy.CFR: 1/3}, None  # Pesos neutros
        except Exception as e:
            logging.exception("Error inesperado en analyze_context")
            return {Strategy.MODEL: 1/3, Strategy.MCTS: 1/3, Strategy.CFR: 1/3}, None  # Pesos neutros

    def _adjust_weights_for_aggression(self, context_weights: Dict[Strategy, float], aggression_factor: float) -> Dict[Strategy, float]:
        """
        Ajusta los pesos basados en el factor de agresión.
        """
        context_weights[Strategy.MODEL] += aggression_factor * 0.1
        context_weights[Strategy.MCTS] -= aggression_factor * 0.05
        context_weights[Strategy.CFR] -= aggression_factor * 0.05
        return context_weights

    def _adjust_weights_for_stack_size(self, context_weights: Dict[Strategy, float], stack_size_factor: float) -> Dict[Strategy, float]:
        """
        Ajusta los pesos basados en el factor del tamaño del stack.
        """
        context_weights[Strategy.MODEL] += stack_size_factor * 0.05
        context_weights[Strategy.MCTS] += stack_size_factor * 0.025
        context_weights[Strategy.CFR] -= stack_size_factor * 0.075
        return context_weights

    def _adjust_weights_for_position(self, context_weights: Dict[Strategy, float], position_factor: float) -> Dict[Strategy, float]:
        """
        Ajusta los pesos basados en el factor de la posición.
        """
        context_weights[Strategy.MODEL] -= position_factor * 0.025
        context_weights[Strategy.MCTS] += position_factor * 0.05
        context_weights[Strategy.CFR] -= position_factor * 0.025
        return context_weights

    def _adjust_weights_for_round_phase(self, context_weights: Dict[Strategy, float], round_phase_factor: float) -> Dict[Strategy, float]:
        """
        Ajusta los pesos basados en el factor de la fase de la ronda.
        """
        context_weights[Strategy.MODEL] += round_phase_factor * 0.03
        context_weights[Strategy.MCTS] -= round_phase_factor * 0.015
        context_weights[Strategy.CFR] -= round_phase_factor * 0.015
        return context_weights

    def _normalize_weights(self, context_weights: Dict[Strategy, float]) -> Dict[Strategy, float]:
        """
        Normaliza los pesos para que sumen 1.0.
        """
        total = sum(context_weights.values())
        if total > 0:
            scaling_factor = 1.0 / total
            context_weights = {k: v * scaling_factor for k, v in context_weights.items()}
        else:
            # Manejar el caso donde la suma es cero
            context_weights = {k: 1.0 / len(context_weights) for k in context_weights}  # Pesos iguales

        # Comprobar la suma total
        total = sum(context_weights.values())
        if abs(total - 1.0) > 1e-9:  # Tolerancia para errores de punto flotante
            logging.warning(f"La suma de los pesos no es 1.0 (es {total}).")

        return context_weights

    def _validate_round_state(self, round_state: Dict):
        """
        Valida el estado de la ronda.
        """
        required_keys = ["stack_sizes", "current_player_position", "round_phase", "actions"]
        for key in required_keys:
            if key not in round_state:
                raise InvalidRoundStateError(f"Falta la clave '{key}' en round_state.")
            if not isinstance(round_state[key], (list, int, str)):
                raise InvalidRoundStateError(f"El tipo de '{key}' es incorrecto.")

        if not isinstance(round_state["stack_sizes"], list) or len(round_state["stack_sizes"]) < 2:
            raise InvalidRoundStateError("stack_sizes debe ser una lista con al menos 2 valores.")

        if not all(isinstance(size, (int, float)) and size >= 0 for size in round_state["stack_sizes"]):
            raise InvalidRoundStateError("stack_sizes debe contener valores numéricos no negativos.")

        if round_state["round_phase"] not in ["PREFLOP", "FLOP", "TURN", "RIVER"]:
            raise InvalidRoundStateError("round_phase debe ser uno de los valores permitidos.")

        if not isinstance(round_state["current_player_position"], int) or not 0 <= round_state["current_player_position"] < self.num_players:
            raise InvalidRoundStateError(f"La posición debe estar entre 0 y {self.num_players - 1}.")

    def _calculate_aggression_factor(self) -> float:
        """
        Calcula un factor de agresión basado en el historial de acciones.

        El factor se calcula como la proporción de acciones agresivas (apuestas y raises)
        en el historial reciente. Un valor más alto indica mayor agresión.

        Este factor aumenta el peso de la estrategia MODEL y disminuye el peso de MCTS y CFR.
        """
        if not self.historical_data:
            return 0.0

        # Ejemplo: Calcular la frecuencia de apuestas y raises en el historial reciente
        aggressive_actions = 0
        total_actions = 0
        for round_data in self.historical_data:
            if "actions" in round_data.get("actions", []):
                for action in round_data["actions"]:
                    total_actions += 1
                    if action in ["BET", "RAISE"]:
                        aggressive_actions += 1

        if total_actions == 0:
            return 0.0

        return aggressive_actions / total_actions

    def _calculate_stack_size_factor(self, stack_sizes: List[float]) -> float:
        """
        Calcula un factor basado en el tamaño del stack.
        """
        # Validar que stack_sizes contenga al menos 2 valores
        if not isinstance(stack_sizes, list) or len(stack_sizes) < 2:
            logging.warning("stack_sizes debe ser una lista con al menos 2 valores. Devolviendo factor neutro.")
            return 0.0

        if all(size == 0 for size in stack_sizes):
            return 0.0  # Todos los stacks son cero

        # Normalizar los tamaños de stack
        max_stack = max(stack_sizes)
        normalized_stacks = [size / max_stack for size in stack_sizes]
        
        # Calcular la diferencia entre el stack más grande y el más pequeño
        stack_difference = max(normalized_stacks) - min(normalized_stacks)
        
        # Devolver un factor basado en la diferencia
        return stack_difference

    def _calculate_position_factor(self, position: int) -> float:
        """
        Calcula un factor basado en la posición en la mesa.
        """
        # Validar que la posición esté dentro del rango válido
        if not isinstance(position, int) or not 0 <= position < self.num_players:
            logging.warning(f"La posición debe estar entre 0 y {self.num_players - 1}. Devolviendo factor neutro.")
            return 0.0

        # Asumiendo una mesa de 9 jugadores
        if position < 3:  # Early position
            return -0.2
        elif position < 6:  # Middle position
            return 0.0
        else:  # Late position
            return 0.2
    
    def _calculate_round_phase_factor(self, round_phase: str) -> float:
        """
        Calcula un factor basado en la fase de la ronda.
        """
        if round_phase == "PREFLOP":
            return -0.1
        elif round_phase == "FLOP":
            return 0.05
        elif round_phase == "TURN":
            return 0.1
        elif round_phase == "RIVER":
            return 0.15
        else:
            return 0.0

    def _run_self_test(self):
        """
        Función de prueba interna para verificar el comportamiento del analizador.
        """
        print("Iniciando self-test...")

        # Escenario 1: Agresión alta, stacks similares, early position, preflop
        round_state_1 = {
            "stack_sizes": [100, 100, 100, 100, 100, 100, 100, 100, 100],
            "current_player_position": 0,
            "round_phase": "PREFLOP",
            "actions": ["RAISE", "RAISE", "CALL"]
        }
        weights_1, factors_1 = self.analyze_context([], round_state_1)
        print(f"Escenario 1: Pesos = {weights_1}, Factores = {factors_1}")

        # Escenario 2: Agresión baja, stacks diferentes, late position, river
        round_state_2 = {
            "stack_sizes": [50, 50, 50, 50, 50, 50, 50, 50, 200],
            "current_player_position": 8,
            "round_phase": "RIVER",
            "actions": ["CHECK", "CHECK", "CHECK"]
        }
        weights_2, factors_2 = self.analyze_context([], round_state_2)
        print(f"Escenario 2: Pesos = {weights_2}, Factores = {factors_2}")

        # Escenario 3: Inputs inválidos
        round_state_3 = {
            "stack_sizes": [100],
            "current_player_position": 10,
            "round_phase": "INVALID",
            "actions": []
        }
        weights_3, factors_3 = self.analyze_context([], round_state_3)
        print(f"Escenario 3: Pesos = {weights_3}, Factores = {factors_3}")

        print("Self-test completado.")

class TestGameContextAnalyzer(unittest.TestCase):
    """
    Pruebas unitarias para GameContextAnalyzer.
    """
    def setUp(self):
        self.analyzer = GameContextAnalyzer()

    def test_valid_round_state(self):
        round_state = {
            "stack_sizes": [100, 100],
            "current_player_position": 0,
            "round_phase": "PREFLOP",
            "actions": ["BET"]
        }
        try:
            self.analyzer._validate_round_state(round_state)
        except InvalidRoundStateError:
            self.fail("Valid round_state raised an exception.")

    def test_invalid_round_state(self):
        round_state = {
            "stack_sizes": [100],
            "current_player_position": 10,
            "round_phase": "INVALID",
            "actions": []
        }
        with self.assertRaises(InvalidRoundStateError):
            self.analyzer._validate_round_state(round_state)

    def test_calculate_aggression_factor(self):
        # Configurar un historial de datos de ejemplo
        self.analyzer.historical_data = deque([
            {"actions": ["BET", "RAISE"]},
            {"actions": ["CHECK", "CALL"]},
            {"actions": ["RAISE", "FOLD"]}
        ])
        # Calcular el factor de agresión
        aggression_factor = self.analyzer._calculate_aggression_factor()
        # Verificar que el factor de agresión sea el esperado
        self.assertEqual(aggression_factor, 3/8)

    def test_calculate_stack_size_factor(self):
        # Proporcionar tamaños de stack de ejemplo
        stack_sizes = [50, 100, 150]
        # Calcular el factor de tamaño del stack
        stack_size_factor = self.analyzer._calculate_stack_size_factor(stack_sizes)
        # Verificar que el factor de tamaño del stack sea el esperado
        self.assertAlmostEqual(stack_size_factor, 1.0, places=7)

# Ejecutar self-test si el script se ejecuta directamente
if __name__ == "__main__":
    # Ejemplo de uso con archivo de configuración
    analyzer = GameContextAnalyzer(config_path="config.yaml")
    analyzer._run_self_test()

    # Ejecutar pruebas unitarias
    unittest.main()