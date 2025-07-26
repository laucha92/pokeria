from collections import deque
from typing import List, Dict, Tuple, Optional
from enum import Enum

class Strategy(Enum):
    MODEL = "model"
    MCTS = "mcts"
    CFR = "cfr"

class GameContextAnalyzer:
    """
    Analiza el contexto del juego para ajustar los pesos de las estrategias.
    """
    def __init__(self, num_players: int = 9, historical_data: deque = deque(maxlen=10), verbose: bool = False,
                 min_stack_size: float = 10.0, max_aggression_threshold: float = 0.8):
        """
        Inicializa el analizador de contexto del juego.

        Args:
            num_players (int): Número de jugadores en la mesa.
            historical_data (deque): Historial de datos del juego.
            verbose (bool): Modo debug para imprimir valores de los factores.
            min_stack_size (float): Límite mínimo para el tamaño del stack.
            max_aggression_threshold (float): Umbral máximo para el factor de agresión.
        """
        self.num_players = num_players
        self.historical_data = historical_data
        self.verbose = verbose
        self.min_stack_size = min_stack_size
        self.max_aggression_threshold = max_aggression_threshold

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
        stack_size_factor = self._calculate_stack_size_factor(round_state.get("stack_sizes", []))
        position_factor = self._calculate_position_factor(round_state.get("current_player_position", -1))
        round_phase_factor = self._calculate_round_phase_factor(round_state.get("round_phase", ""))

        # Ajustar pesos basados en los factores
        # Agresión: Aumenta el peso del modelo si hay agresión reciente.
        context_weights[Strategy.MODEL] += aggression_factor * 0.1
        context_weights[Strategy.MCTS] -= aggression_factor * 0.05
        context_weights[Strategy.CFR] -= aggression_factor * 0.05

        # Tamaño del stack: Ajusta los pesos según el tamaño del stack.
        context_weights[Strategy.MODEL] += stack_size_factor * 0.05
        context_weights[Strategy.MCTS] += stack_size_factor * 0.025
        context_weights[Strategy.CFR] -= stack_size_factor * 0.075

        # Posición en la mesa: Ajusta los pesos según la posición del jugador.
        context_weights[Strategy.MODEL] -= position_factor * 0.025
        context_weights[Strategy.MCTS] += position_factor * 0.05
        context_weights[Strategy.CFR] -= position_factor * 0.025
        
        # Fase de la ronda: Ajusta los pesos según la fase de la ronda.
        context_weights[Strategy.MODEL] += round_phase_factor * 0.03
        context_weights[Strategy.MCTS] -= round_phase_factor * 0.015
        context_weights[Strategy.CFR] -= round_phase_factor * 0.015

        # Normalización final de pesos
        total = sum(context_weights.values())
        if total > 0:
            context_weights = {k: v / total for k, v in context_weights.items()}
        else:
            # Manejar el caso donde la suma es cero
            context_weights = {k: 1.0 / len(context_weights) for k in context_weights}  # Pesos iguales

        # Comprobar la suma total
        total = sum(context_weights.values())
        if abs(total - 1.0) > 1e-9:  # Tolerancia para errores de punto flotante
            print(f"Warning: La suma de los pesos no es 1.0 (es {total}).")

        # Registrar factores si verbose es True
        factors = None
        if self.verbose:
            factors = {
                "aggression_factor": aggression_factor,
                "stack_size_factor": stack_size_factor,
                "position_factor": position_factor,
                "round_phase_factor": round_phase_factor
            }
            print(f"Factores: {factors}")

        return context_weights, factors

    def _calculate_aggression_factor(self) -> float:
        """
        Calcula un factor de agresión basado en el historial de acciones.
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
            print("Warning: stack_sizes debe ser una lista con al menos 2 valores. Devolviendo factor neutro.")
            return 0.0

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
            print(f"Warning: La posición debe estar entre 0 y {self.num_players - 1}. Devolviendo factor neutro.")
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

# Ejecutar self-test si el script se ejecuta directamente
if __name__ == "__main__":
    analyzer = GameContextAnalyzer(num_players=9, verbose=True)
    analyzer._run_self_test()
