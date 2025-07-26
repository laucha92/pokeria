from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class FallbackPolicy(ABC):
    """
    Clase base abstracta para una política de respaldo (fallback).
    """

    @abstractmethod
    def get_fallback_strategy(self, num_actions: int, round_state: Dict[str, Any]) -> np.ndarray:
        """
        Devuelve la estrategia de respaldo para un estado de ronda dado.

        Args:
            num_actions (int): Número de acciones disponibles (ej. fold, call, raise...).
            round_state (dict): Estado de la ronda actual.

        Returns:
            np.ndarray: Vector de probabilidades de acciones, normalizado.
        """
        pass


class UniformFallbackPolicy(FallbackPolicy):
    """
    Política de respaldo que distribuye uniformemente la probabilidad entre todas las acciones.
    """

    def get_fallback_strategy(self, num_actions: int, round_state: Dict[str, Any]) -> np.ndarray:
        return np.ones(num_actions) / num_actions


class RandomFallbackPolicy(FallbackPolicy):
    """
    Política de respaldo que genera una distribución aleatoria normalizada.
    """

    def get_fallback_strategy(self, num_actions: int, round_state: Dict[str, Any]) -> np.ndarray:
        probs = np.random.rand(num_actions)
        return probs / probs.sum()


class HeuristicFallbackPolicy(FallbackPolicy):
    """
    Política de respaldo basada en heurística según la fuerza de la mano.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Inicializa la política heurística.

        Args:
            threshold (float): Umbral de fuerza de mano para actuar agresivamente.
        """
        self.threshold = threshold

    def get_fallback_strategy(self, num_actions: int, round_state: Dict[str, Any]) -> np.ndarray:
        if "hand_strength" not in round_state:
            raise ValueError("El estado de la ronda debe incluir 'hand_strength'.")

        strength = round_state["hand_strength"]
        strategy = np.zeros(num_actions)

        if num_actions < 2:
            raise ValueError("Se requieren al menos dos acciones para usar esta política.")

        if strength >= self.threshold:
            # Agresivo: poco fold, más apuestas
            strategy[0] = 0.1  # fold
            strategy[1:] = 0.9 / (num_actions - 1)
        else:
            # Conservador: mucho fold
            strategy[0] = 0.8  # fold
            strategy[1:] = 0.2 / (num_actions - 1)

        return strategy
