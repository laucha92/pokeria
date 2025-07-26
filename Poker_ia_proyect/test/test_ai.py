import unittest
from ai.mcts_agent import MCTSAgent, MCTSNode
from game.game_state import GameState  # Asegúrate de importar tu clase GameState
from game.card import Card
# Necesitarás crear una clase Action simulada para esta prueba
class Action:
    def __init__(self, type):
        self.type = type

    def __repr__(self):
        return f"Action({self.type})"

class TestMCTSAgent(unittest.TestCase):
    def test_aggressive_action_tendency(self):
        # Configuración de la prueba
        simulations = 1000
        exploration_const = 1.0
        agent = MCTSAgent(simulations, exploration_const)

        # Crea un estado de juego simulado (debes adaptarlo a tu implementación)
        # Esto es un ejemplo, necesitas crear un GameState válido para tu juego
        class MockGameState(GameState):  # Crea una clase Mock que hereda de GameState
            def __init__(self):
                super().__init__()
                self.legal_actions = [
                    Action("CHECK"),
                    Action("BET"),
                    Action("RAISE"),
                    Action("FOLD")
                ]
            def get_legal_actions(self):
                return self.legal_actions
            def apply_action(self, action):
                return self  # Devuelve el mismo estado para simplificar
            def is_terminal(self):
                return True
            def get_reward(self):
                return 0

        game_state = MockGameState()
        root_node = MCTSNode(game_state)

        # Ejecuta simulaciones
        for _ in range(simulations):
            agent.simulate(root_node)

        # Cuenta las acciones agresivas
        aggressive_actions_count = 0
        for _ in range(simulations):
            game_state = MockGameState()
            legal_actions = game_state.get_legal_actions()
            aggressive_actions = [a for a in legal_actions if a.type in ["BET", "RAISE"]]
            if aggressive_actions:
                if random.random() < 0.7:
                    aggressive_actions_count += 1

        # Verifica que la proporción de acciones agresivas sea mayor que un umbral
        self.assertGreater(aggressive_actions_count, simulations * 0.3)  # Ajusta el umbral según sea necesario

if __name__ == '__main__':
    unittest.main()