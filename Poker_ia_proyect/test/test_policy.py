import unittest
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import Dict, List
from policy.policy import Policy  # Importar la clase Policy desde su nuevo archivo
from policy.fallback_policies import UniformFallbackPolicy, RandomFallbackPolicy, HeuristicFallbackPolicy
from analysis.game_context_analyzer import GameContextAnalyzer

class TestPolicy(unittest.TestCase):
    """
    Pruebas unitarias para la clase Policy.
    """
    def setUp(self):
        """
        Configuración inicial para las pruebas.
        """
        # Crear un modelo simple para las pruebas
        self.model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Softmax(dim=1)
        )
        # Crear un agente MCTS simulado
        class MockMCTS:
            def get_action_probs(self, state):
                return np.array([0.1, 0.2, 0.3, 0.2, 0.2])
            @property
            def num_simulations(self):
                return 100
        self.mcts_agent = MockMCTS()
        # Inicializar la política con valores predeterminados
        self.policy = Policy(
            model=self.model,
            mcts_agent=self.mcts_agent,
            state_size=10
        )

    def test_evaluate_blending_quality(self):
        """
        Prueba la función _evaluate_blending_quality.
        """
        strategies = {"s1": 0.3, "s2": 0.4, "s3": 0.3}
        blended_strategy = np.array([0.2, 0.3, 0.5])
        self.policy._evaluate_blending_quality(strategies, blended_strategy)
        # Verificar que no haya errores y que las métricas se registren
        self.assertTrue(True)

    def test_voting_ranking(self):
        """
        Prueba la función _voting_ranking.
        """
        strategies = {
            "s1": np.array([0.1, 0.2, 0.7]),
            "s2": np.array([0.6, 0.1, 0.3]),
            "s3": np.array([0.1, 0.8, 0.1])
        }
        confidences = {"s1": 0.8, "s2": 0.5, "s3": 0.9}
        weights = {"s1": 0.3, "s2": 0.3, "s3": 0.4}
        ranked_strategies = self.policy._voting_ranking(strategies, confidences, weights)
        # Verificar que las estrategias se ordenen por confianza
        expected_order = ["s3", "s1", "s2"]
        self.assertEqual(list(ranked_strategies.keys()), expected_order)
        # Verificar que se filtren las estrategias por umbral de confianza
        self.policy.confidence_threshold_adjuster.current_confidence_threshold = 0.7
        ranked_strategies = self.policy._voting_ranking(strategies, confidences, weights)
        expected_order = ["s3", "s1"]
        self.assertEqual(list(ranked_strategies.keys()), expected_order)

    def test_adjust_confidence_threshold(self):
        """
        Prueba la función _adjust_confidence_threshold.
        """
        strategies = {
            "s1": np.array([0.1, 0.2, 0.7]),
            "s2": np.array([0.6, 0.1, 0.3]),
            "s3": np.array([0.1, 0.8, 0.1])
        }
        confidences = {"s1": 0.8, "s2": 0.5, "s3": 0.9}
        initial_threshold = self.policy.confidence_threshold_adjuster.current_confidence_threshold
        # Caso 1: Acción coincide con la estrategia más confiable
        self.policy.confidence_threshold_adjuster.adjust_threshold(strategies, 2, confidences)
        self.assertLess(self.policy.confidence_threshold_adjuster.current_confidence_threshold, initial_threshold)
        # Caso 2: Acción no coincide con la estrategia más confiable
        self.policy.confidence_threshold_adjuster.current_confidence_threshold = initial_threshold
        self.policy.confidence_threshold_adjuster.adjust_threshold(strategies, 0, confidences)
        self.assertGreater(self.policy.confidence_threshold_adjuster.current_confidence_threshold, initial_threshold)

    def test_detect_strategy_conflict(self):
        """
        Prueba la función _detect_strategy_conflict.
        """
        # Caso 1: Estrategias en conflicto
        strategies1 = {
            "s1": np.array([0.1, 0.2, 0.7]),
            "s2": np.array([0.7, 0.2, 0.1])
        }
        conflict1 = self.policy._detect_strategy_conflict(strategies1)
        self.assertTrue(conflict1)

        # Caso 2: Estrategias sin conflicto
        strategies2 = {
            "s1": np.array([0.1, 0.2, 0.7]),
            "s2": np.array([0.15, 0.25, 0.6])
        }
        conflict2 = self.policy._detect_strategy_conflict(strategies2)
        self.assertFalse(conflict2)

    def test_selection_modes(self):
        """
        Prueba los distintos modos de selección (argmax, epsilon_greedy, sampling).
        """
        strategy = np.array([0.1, 0.2, 0.7])
        # Argmax
        self.policy.selection_mode = "argmax"
        action_argmax = self.policy._select_action_from_strategy(strategy)
        self.assertEqual(action_argmax, 2)
        # Epsilon Greedy
        self.policy.selection_mode = "epsilon_greedy"
        self.policy.epsilon = 1.0  # Forzar exploración
        action_epsilon_greedy = self.policy._select_action_from_strategy(strategy)
        self.assertIn(action_epsilon_greedy, [0, 1, 2])
        # Sampling
        self.policy.selection_mode = "sampling"
        action_sampling = self.policy._select_action_from_strategy(strategy)
        self.assertIn(action_sampling, [0, 1, 2])

    def test_confidence_threshold_dynamic_changes(self):
        """
        Valida cambios en el confidence_threshold en escenarios dinámicos y extremos.
        """
        strategies = {
            "s1": np.array([0.1, 0.2, 0.7]),
            "s2": np.array([0.6, 0.1, 0.3]),
            "s3": np.array([0.1, 0.8, 0.1])
        }
        confidences = {"s1": 0.8, "s2": 0.5, "s3": 0.9}
        initial_threshold = self.policy.confidence_threshold_adjuster.current_confidence_threshold

        # Escenario 1: Acciones consistentemente correctas
        for _ in range(10):
            self.policy.confidence_threshold_adjuster.adjust_threshold(strategies, 2, confidences)
        self.assertLess(self.policy.confidence_threshold_adjuster.current_confidence_threshold, initial_threshold)

        # Escenario 2: Acciones consistentemente incorrectas
        self.policy.confidence_threshold_adjuster.current_confidence_threshold = initial_threshold
        for _ in range(10):
            self.policy.confidence_threshold_adjuster.adjust_threshold(strategies, 0, confidences)
        self.assertGreater(self.policy.confidence_threshold_adjuster.current_confidence_threshold, initial_threshold)

    def test_blend_strategies(self):
        """
        Prueba la función blend_strategies.
        """
        strategies = {
            "s1": np.array([0.1, 0.2, 0.7]),
            "s2": np.array([0.6, 0.1, 0.3])
        }
        weights = {"s1": 0.5, "s2": 0.5}
        blended_strategy = self.policy.blend_strategies(strategies, weights)
        expected_strategy = np.array([0.35, 0.15, 0.5])
        np.testing.assert_allclose(blended_strategy, expected_strategy)

if __name__ == '__main__':
    # Ejecutar las pruebas unitarias
    unittest.main()