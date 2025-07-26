import math
import random
import time
from typing import List, Dict, Callable, Optional
from game.game_state import GameState

class MCTSNode:
    """
    Representa un nodo en el árbol de búsqueda MCTS.
    """
    def __init__(self, game_state: GameState, parent=None, action=None):
        self.game_state = game_state
        self.parent = parent
        self.action = action  # Acción que llevó a este nodo desde el padre
        self.children = {}  # {action: MCTSNode}
        self.visits = 0
        # Wins ahora es un diccionario por jugador
        self.wins: Dict[int, float] = {}  # {player_id: score}

    def is_terminal(self):
        """
        Verifica si el nodo representa un estado terminal del juego.
        """
        return self.game_state.is_terminal()

    def is_fully_expanded(self, agent):
        """
        Verifica si el nodo ha sido completamente expandido (todos los movimientos legales explorados).
        """
        return len(self.children) == len(agent.get_legal_actions(self.game_state))

    def ucb1(self, exploration_const: float, player_id: int):
        """
        Calcula el valor UCB1 para este nodo desde la perspectiva de un jugador.
        """
        if self.visits == 0:
            return float('inf')  # Priorizar nodos no visitados
        # Obtiene el score del jugador o 0 si no existe
        player_wins = self.wins.get(player_id, 0)
        return (player_wins / self.visits) + exploration_const * math.sqrt(math.log(self.parent.visits) / self.visits)

class MCTSAgent:
    """
    Agente que utiliza Monte Carlo Tree Search (MCTS) para tomar decisiones.
    """
    def __init__(self, simulations: int, exploration_const: float,
                 time_limit: float = None,  # Límite de tiempo opcional
                 simulation_policy: Callable[[GameState], Dict[int, float]] = None,  # Política de simulación configurable
                 max_depth: Optional[int] = None):  # Profundidad máxima opcional
        """
        Inicializa el agente MCTS.

        Args:
            simulations (int): Número de simulaciones a realizar por cada búsqueda MCTS.
            exploration_const (float): Constante de exploración para UCB1.
            time_limit (float, optional): Límite de tiempo en segundos para la búsqueda MCTS. Defaults to None.
            simulation_policy (Callable[[GameState], Dict[int, float]], optional): Política de simulación configurable. Defaults to None.
            max_depth (int, optional): Límite de profundidad para la búsqueda. Defaults to None.
        """
        self.simulations = simulations
        self.exploration_const = exploration_const
        self.time_limit = time_limit
        self.simulation_policy = simulation_policy or self._default_simulation_policy  # Usa la política proporcionada o la predeterminada
        self.max_depth = max_depth

    def select(self, node: MCTSNode, player_id: int, depth: int = 0) -> MCTSNode:
        """
        Selecciona un nodo para expandir utilizando UCB1.

        Args:
            node (MCTSNode): Nodo desde el cual comenzar la selección.
            player_id (int): ID del jugador para calcular UCB1.
            depth (int): Profundidad actual de la búsqueda.

        Returns:
            MCTSNode: Nodo seleccionado para expandir.
        """
        while not node.is_terminal():
            if self.max_depth is not None and depth >= self.max_depth:
                return node  # Limite de profundidad alcanzado
            if not node.is_fully_expanded(self):
                return self.expand(node)
            else:
                # Seleccionar el hijo con el valor UCB1 más alto
                best_child = None
                best_ucb1 = float('-inf')  # Inicializa con infinito negativo
                candidates = list(node.children.values())
                random.shuffle(candidates) #Shuffle para evitar sesgos
                for child in candidates:
                    ucb1_value = child.ucb1(self.exploration_const, player_id)
                    if ucb1_value > best_ucb1:
                        best_ucb1 = ucb1_value
                        best_child = child
                if best_child is None:
                    # Todos los hijos tienen UCB1 infinito (poco probable, pero posible)
                    # Elige uno al azar para evitar un bucle infinito
                    best_child = random.choice(candidates)
                node = best_child
                depth += 1
        return node

    def expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expande un nodo creando nuevos nodos hijo para cada acción legal no explorada.

        Args:
            node (MCTSNode): Nodo a expandir.

        Returns:
            MCTSNode: Un nuevo nodo hijo (aleatorio)
        """
        legal_actions = self.get_legal_actions(node.game_state)
        #Verifica si hay acciones legales
        if not legal_actions:
            return node #Retorna el nodo actual si no hay acciones legales

        # Encuentra las acciones que aún no han sido exploradas
        unexplored_actions = [action for action in legal_actions if action not in node.children]

        # Elige una acción no explorada al azar
        action = random.choice(unexplored_actions)
        # Crea un nuevo estado del juego aplicando la acción
        new_game_state = node.game_state.apply_action(action)
        # Crea un nuevo nodo hijo
        new_node = MCTSNode(new_game_state, parent=node, action=action)
        # Agrega el nuevo nodo a los hijos del nodo actual
        node.children[action] = new_node
        return new_node

    def simulate(self, node: MCTSNode) -> Dict[int, float]:
        """
        Simula un juego aleatorio desde un nodo dado hasta un estado terminal,
        utilizando la política de simulación configurada.

        Args:
            node (MCTSNode): Nodo desde el cual comenzar la simulación.

        Returns:
            Dict[int, float]: Recompensas obtenidas al final del juego simulado para cada jugador.
        """
        return self.simulation_policy(node.game_state)

    def _default_simulation_policy(self, game_state: GameState, depth: int = 0) -> Dict[int, float]:
        """
        Política de simulación predeterminada que prioriza acciones agresivas.

        Args:
            game_state (GameState): El estado actual del juego.
            depth (int): Profundidad actual de la simulación.

        Returns:
            Dict[int, float]: Recompensas obtenidas al final del juego simulado para cada jugador.
        """
        if self.max_depth is not None and depth >= self.max_depth:
            return game_state.get_rewards()  # Devuelve las recompensas actuales si se alcanza la profundidad máxima

        game_state = game_state.copy()
        while not game_state.is_terminal():
            legal_actions = self.get_legal_actions(game_state)
            #Verifica si hay acciones legales
            if not legal_actions:
                return {player_id: 0.0 for player_id in game_state.get_player_ids()}  #Retorna 0 para todos los jugadores si no hay acciones legales

            # Priorizar acciones agresivas (apostar, subir)
            aggressive_actions = [a for a in legal_actions if a.type in ["BET", "RAISE"]]  # Asume que tus acciones tienen un atributo 'type'
            if aggressive_actions:
                # Elige una acción agresiva al azar con cierta probabilidad
                if random.random() < 0.7:  # 70% de probabilidad de elegir una acción agresiva
                    action = random.choice(aggressive_actions)
                else:
                    action = random.choice(legal_actions)
            else:
                action = random.choice(legal_actions)
            game_state = game_state.apply_action(action)
            depth += 1
            if self.max_depth is not None and depth >= self.max_depth:
                break #Sale del bucle si se alcanza la profundidad máxima
        return game_state.get_rewards()

    def backpropagate(self, node: MCTSNode, rewards: Dict[int, float]):
        """
        Propaga el resultado de la simulación hacia arriba en el árbol.

        Args:
            node (MCTSNode): Nodo desde el cual comenzar la propagación.
            rewards (Dict[int, float]): Recompensas obtenidas al final de la simulación para cada jugador.
        """
        while node is not None:
            node.visits += 1
            # Acumula las recompensas para cada jugador
            for player_id, reward in rewards.items():
                node.wins[player_id] = node.wins.get(player_id, 0) + reward
            node = node.parent

    def best_action(self, game_state: GameState, player_id: int):
        """
        Obtiene la mejor acción basada en la búsqueda MCTS.

        Args:
            game_state (GameState): Estado actual del juego.
            player_id (int): ID del jugador que está tomando la decisión.

        Returns:
            acción: La mejor acción a tomar.

        Ejemplo de uso:
            # Asumiendo que tienes un objeto game_state y el ID del jugador actual
            action = agent.best_action(game_state, player_id=1)
        """
        root = MCTSNode(game_state)
        start_time = time.time()
        num_simulations = 0

        while True:
            # Verifica si se ha alcanzado el límite de tiempo
            if self.time_limit is not None and time.time() - start_time > self.time_limit:
                break
            if num_simulations >= self.simulations:
                break

            # Fase 1: Selección
            leaf = self.select(root, player_id)
            # Fase 2: Simulación
            rewards = self.simulate(leaf)
            # Fase 3: Backpropagation
            self.backpropagate(leaf, rewards)

            num_simulations += 1

        # Después de las simulaciones, elige la acción con el mejor criterio
        best_action = None
        best_value = float('-inf')

        for action, child in root.children.items():
            # Criterio: Mayor promedio de recompensa para el jugador actual
            player_reward = child.wins.get(player_id, 0) / (child.visits if child.visits > 0 else 1)
            if player_reward > best_value:
                best_value = player_reward
                best_action = action

        if best_action is None:
            # Si no se ha explorado ninguna acción, elige una al azar
            legal_actions = self.get_legal_actions(game_state)
            if legal_actions:
                best_action = random.choice(legal_actions)
            else:
                return None  # No hay acciones posibles

        return best_action

    def get_legal_actions(self, game_state: GameState) -> List:
        """
        Obtiene las acciones legales para el estado del juego.

        Args:
            game_state (GameState): El estado actual del juego.

        Returns:
            List: Una lista de acciones legales.
        """
        legal_actions = game_state.get_legal_actions()
        return legal_actions