from collections import deque
from typing import List, Dict

class GameContextAnalyzer:
    """
    Analiza el contexto del juego para ajustar los pesos de las estrategias.
    """
    def __init__(self, historical_data: deque = deque(maxlen=10)):
        self.historical_data = historical_data

    def analyze_context(self, game_state: List[float], round_state: Dict) -> Dict[str, float]:
        """
        Analiza el estado del juego y devuelve un diccionario de pesos ajustados.
        """
        # Lógica de análisis de contexto (ejemplo)
        context_weights = {
            "model": 0.3,
            "mcts": 0.4,
            "cfr": 0.3
        }
        # Ajustar pesos basados en el historial de agresión
        aggression_factor = self._calculate_aggression_factor()
        context_weights["model"] += aggression_factor * 0.1
        context_weights["mcts"] -= aggression_factor * 0.05
        context_weights["cfr"] -= aggression_factor * 0.05

        # Ajustar pesos basados en el tamaño del stack
        stack_size_factor = self._calculate_stack_size_factor(round_state["stack_sizes"])
        context_weights["model"] += stack_size_factor * 0.05
        context_weights["mcts"] += stack_size_factor * 0.025
        context_weights["cfr"] -= stack_size_factor * 0.075

        return context_weights

    def _calculate_aggression_factor(self) -> float:
        """
        Calcula un factor de agresión basado en el historial de acciones.
        """
        # Lógica para calcular el factor de agresión
        return 0.5

    def _calculate_stack_size_factor(self, stack_sizes: List[float]) -> float:
        """
        Calcula un factor basado en el tamaño del stack.
        """
        # Lógica para calcular el factor del tamaño del stack
        return 0.2