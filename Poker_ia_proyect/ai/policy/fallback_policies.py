import numpy as np
from typing import Dict

class FallbackPolicy:
    """
    Clase base para políticas de fallback.
    """
    def get_fallback_strategy(self, num_actions: int, round_state: Dict) -> np.ndarray:
        """
        Devuelve una estrategia de fallback.
        """
        raise NotImplementedError

class UniformFallbackPolicy(FallbackPolicy):
    """
    Política de fallback uniforme.
    """
    def get_fallback_strategy(self, num_actions: int, round_state: Dict) -> np.ndarray:
        """
        Devuelve una estrategia uniforme.
        """
        return np.ones(num_actions) / num_actions

class RandomFallbackPolicy(FallbackPolicy):
    """
    Política de fallback aleatoria.
    """
    def get_fallback_strategy(self, num_actions: int, round_state: Dict) -> np.ndarray:
        """
        Devuelve una estrategia aleatoria.
        """
        strategy = np.random.rand(num_actions)
        return strategy / np.sum(strategy)

class HeuristicFallbackPolicy(FallbackPolicy):
    """
    Política de fallback basada en heurísticas tácticas.
    """
    def get_fallback_strategy(self, num_actions: int, round_state: Dict) -> np.ndarray:
        """
        Devuelve una estrategia basada en heurísticas tácticas.
        """
        # Lógica para implementar una estrategia tight/aggressive conservadora
        strategy = np.zeros(num_actions)
        # Ejemplo: fold con manos débiles, apostar con manos fuertes
        if round_state["hand_strength"] < 0.3:
            strategy[0] = 0.9  # Fold
            strategy[1:] = 0.1 / (num_actions - 1)
        else:
            strategy[0] = 0.1  # Fold
            strategy[1:] = 0.9 / (num_actions - 1)
        return strategy