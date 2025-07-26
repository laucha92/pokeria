import numpy as np
from typing import Dict, List, Callable, Tuple, Optional
from game.game_state import GameState
from utils.game_utils import infoset_format  # Asumo que tienes esta función
import logging  # Para el registro de progreso
import multiprocessing # Para paralelización a nivel de proceso
import time # Para medir tiempos de ejecución
from collections import defaultdict # Para thread-safe defaultdict
import threading

# Configura el logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CFRError(Exception):
    """Excepción base para errores relacionados con CFR."""
    pass

class InvalidInfosetError(CFRError):
    """Excepción para infosets inválidos."""
    pass

class MissingLegalActionsMethodError(CFRError):
    """Excepción si el modelo no implementa get_legal_actions."""
    pass

class MaxDepthExceededError(CFRError):
    """Excepción para cuando se excede la profundidad máxima de recursión."""
    pass

class CFRTrainer:
    """
    Implementación robusta y optimizada de Counterfactual Regret Minimization (CFR) para juegos de dos jugadores,
    con soporte para paralelización a nivel de proceso y sincronización thread-safe.

    Documentación:
        - El argumento `model` debe ser un objeto que implemente un método `get_legal_actions(infoset)`
          que retorne una lista de acciones legales para un infoset dado.
        - La función `infoset_format(state, history)` debe retornar una cadena que represente el
          conjunto de información del jugador, codificando toda la información relevante del estado
          y el historial de acciones.
    """
    def __init__(self, model, iterations: int, max_depth: int = 50, verbose: bool = False, num_processes: int = 1):
        """
        Inicializa el entrenador CFR.

        Args:
            model: Modelo para almacenar y actualizar estrategias. Debe implementar `get_legal_actions(infoset)`.
            iterations (int): Número de iteraciones de entrenamiento.
            max_depth (int): Profundidad máxima de recursión para evitar stack overflow.
            verbose (bool): Controla la cantidad de mensajes informativos.
            num_processes (int): Número de procesos para paralelizar el entrenamiento.

        Raises:
            MissingLegalActionsMethodError: Si el modelo no implementa `get_legal_actions(infoset)`.
            ValueError: Si iterations, max_depth o num_processes no son válidos.
        """
        if not hasattr(model, 'get_legal_actions') or not callable(model.get_legal_actions):
            raise MissingLegalActionsMethodError("El modelo debe implementar un método 'get_legal_actions(infoset)'")

        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("iterations debe ser un entero positivo.")

        if not isinstance(max_depth, int) or max_depth <= 0:
            raise ValueError("max_depth debe ser un entero positivo.")

        if not isinstance(num_processes, int) or num_processes <= 0:
            raise ValueError("num_processes debe ser un entero positivo.")

        self.model = model
        self.iterations = iterations
        self.max_depth = max_depth
        self.verbose = verbose
        self.num_processes = num_processes
        self.num_players = 2  # Soporte explícito para 2 jugadores

        # Estructuras thread-safe para datos compartidos
        self.regret_sum: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(0))  # {infoset: [regret_per_action]}
        self.strategy: Dict[str, np.ndarray] = {} # {infoset: [strategy_per_action]} (opcional)
        self.strategy_sum: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(0))  # {infoset: [cumulative_strategy]}
        self.lock = threading.Lock() # Lock para proteger el acceso a las estructuras compartidas

    def train(self, game_state: GameState):
        """
        Entrena el algoritmo CFR.

        Args:
            game_state (GameState): Estado inicial del juego.
        """
        if self.num_processes > 1:
            self._train_multiprocess(game_state)
        else:
            self._train_sequential(game_state)

    def _train_sequential(self, game_state: GameState):
        """
        Entrenamiento CFR secuencial (un solo proceso).
        """
        start_time = time.time()
        for i in range(self.iterations):
            if self.verbose and i % 100 == 0:
                elapsed_time = time.time() - start_time
                logging.info(f"Iteración CFR: {i + 1}/{self.iterations}, Tiempo transcurrido: {elapsed_time:.2f}s")  # Registro de progreso
            self.cfr(game_state, [], 1.0, 1.0, depth=0, iteration=i)
        end_time = time.time()
        logging.info(f"Entrenamiento CFR completado en {end_time - start_time:.2f} segundos.")

    def _train_multiprocess(self, game_state: GameState):
        """
        Entrenamiento CFR paralelo (múltiples procesos).
        """
        start_time = time.time()
        pool = multiprocessing.Pool(processes=self.num_processes)
        iterations_per_process = self.iterations // self.num_processes
        remaining_iterations = self.iterations % self.num_processes

        # Crea una lista de argumentos para cada proceso
        args = []
        for i in range(self.num_processes):
            iterations = iterations_per_process + (1 if i < remaining_iterations else 0)
            args.append((game_state, iterations, i))

        # Ejecuta el entrenamiento en paralelo
        pool.starmap(self._cfr_process, args)

        pool.close()
        pool.join()

        end_time = time.time()
        logging.info(f"Entrenamiento CFR paralelo completado en {end_time - start_time:.2f} segundos con {self.num_processes} procesos.")

    def _cfr_process(self, game_state: GameState, iterations: int, process_id: int):
        """
        Función que ejecuta el CFR en un proceso separado.
        """
        start_time = time.time()
        for i in range(iterations):
            iteration = i + process_id * iterations # Calcula el número de iteración global
            if self.verbose and i % 100 == 0:
                elapsed_time = time.time() - start_time
                logging.info(f"Proceso {process_id}: Iteración CFR: {i + 1}/{iterations}, Tiempo transcurrido: {elapsed_time:.2f}s")  # Registro de progreso
            self.cfr(game_state, [], 1.0, 1.0, depth=0, iteration=iteration)
        end_time = time.time()
        logging.info(f"Proceso {process_id} completado en {end_time - start_time:.2f} segundos.")

    def cfr(self, state: GameState, history: List, prob1: float, prob2: float, depth: int, iteration: int) -> Dict[int, float]:
        """
        Implementa el algoritmo CFR recursivamente.

        Args:
            state (GameState): Estado actual del juego.
            history (List): Historial de acciones hasta este punto (puede ser redundante si infoset_format codifica toda la información).
            prob1 (float): Probabilidad de alcanzar este estado para el jugador 1.
            prob2 (float): Probabilidad de alcanzar este estado para el jugador 2.
            depth (int): Profundidad actual de la recursión.
            iteration (int): Número de iteración actual (para métricas/logs).

        Returns:
            Dict[int, float]: Valor esperado para cada jugador en este estado.

        Raises:
            MaxDepthExceededError: Si se excede la profundidad máxima de recursión.
        """
        if depth > self.max_depth:
            raise MaxDepthExceededError(f"Profundidad máxima de recursión alcanzada ({self.max_depth}).")

        if state.is_terminal():
            return state.get_rewards()  # Asume que esto devuelve un dict

        player = state.get_current_player()
        try:
            infoset = infoset_format(state, history)  # Formatea el infoset
            if not isinstance(infoset, str) or not infoset:
                raise InvalidInfosetError(f"infoset_format retornó un infoset inválido: {infoset}")
        except Exception as e:
            raise CFRError(f"Error al obtener o validar el infoset: {e}")

        # Obtiene las acciones legales para este infoset
        try:
            legal_actions = self.model.get_legal_actions(infoset)
            num_actions = len(legal_actions)
            if num_actions == 0:
                logging.warning(f"No hay acciones legales en el infoset: {infoset}")
                return {p: 0.0 for p in state.get_player_ids()}  # Retorna 0 para todos los jugadores
        except Exception as e:
            raise CFRError(f"Error al obtener acciones legales: {e}")

        # Obtiene la estrategia actual para este infoset
        strategy = self._get_strategy(infoset, num_actions)

        # Actualización de estrategia promedio: Llama a update_strategy_sum()
        self._update_strategy_sum(infoset, strategy, prob1 if player == 1 else prob2)

        # Guardar estrategia actual (opcional)
        self.strategy[infoset] = strategy

        expected_value = {p: 0.0 for p in state.get_player_ids()}  # Inicializa el valor esperado

        # Valores de utilidad para cada acción
        action_utils = np.zeros(num_actions)

        # Itera sobre las acciones posibles
        for i, action in enumerate(legal_actions):
            next_state = state.apply_action(action)
            next_history = history + [action]

            # Calcula la probabilidad de alcanzar el siguiente estado
            if player == 1:
                next_prob1 = prob1 * strategy[i]
                next_prob2 = prob2
            else:
                next_prob1 = prob1
                next_prob2 = prob2 * strategy[i]

            # Llama a CFR recursivamente
            rewards = self.cfr(next_state, next_history, next_prob1, next_prob2, depth + 1, iteration)

            # Acumula el valor esperado
            for p, reward in rewards.items():
                expected_value[p] += strategy[i] * reward
            action_utils[i] = rewards[player]

        # Calcula el arrepentimiento (regret)
        regret = self.regret_sum[infoset] # Obtiene el arrepentimiento actual
        regret_new = (action_utils - expected_value[player]) # Calcula el nuevo arrepentimiento
        with self.lock: # Protege el acceso a la estructura compartida
            if regret.shape != regret_new.shape: # Si las dimensiones no coinciden, crea un nuevo array
                self.regret_sum[infoset] = regret_new
            else: # Si las dimensiones coinciden, actualiza el array existente
                self.regret_sum[infoset] += regret_new

        # Métrica simple: Imprime el arrepentimiento promedio cada 100 iteraciones
        if self.verbose and iteration % 100 == 0:
            avg_regret = np.mean(np.abs(self.regret_sum[infoset]))
            logging.info(f"Iteración {iteration}: Arrepentimiento promedio = {avg_regret:.4f} en infoset {infoset}")

        return expected_value

    def _get_strategy(self, infoset: str, num_actions: int) -> np.ndarray:
        """
        Obtiene la estrategia para un conjunto de información dado.

        Args:
            infoset (str): Conjunto de información para el cual obtener la estrategia.
            num_actions (int): Número de acciones legales para este infoset.

        Returns:
            np.ndarray: Estrategia para el infoset (probabilidades para cada acción).
        """
        # Normaliza los arrepentimientos positivos para obtener la estrategia
        regret = self.regret_sum[infoset]
        strategy = np.maximum(regret, 0)
        normalizing_sum = np.sum(strategy)

        if normalizing_sum > 0:
            strategy /= normalizing_sum
        else:
            # Estrategia uniforme si no hay arrepentimiento positivo
            strategy = np.ones(num_actions) / num_actions

        return strategy

    def get_average_strategy(self) -> Dict[str, np.ndarray]:
        """
        Obtiene la estrategia promedio.

        Returns:
            Dict[str, np.ndarray]: Estrategia promedio para cada infoset.
        """
        average_strategy = {}
        for infoset, strategy_sum in self.strategy_sum.items():
            normalizing_sum = np.sum(strategy_sum)
            num_actions = len(strategy_sum) #Asegura que se use la longitud correcta
            if normalizing_sum > 0:
                average_strategy[infoset] = strategy_sum / normalizing_sum
            else:
                # Estrategia uniforme si no hay estrategia acumulada
                average_strategy[infoset] = np.ones(num_actions) / num_actions
        return average_strategy

    def _update_strategy_sum(self, infoset: str, strategy: np.ndarray, prob: float):
        """
        Actualiza la estrategia acumulada.

        Args:
            infoset (str): Conjunto de información.
            strategy (np.ndarray): Estrategia actual.
            prob (float): Probabilidad de alcanzar este infoset.
        """
        with self.lock: # Protege el acceso a la estructura compartida
            strategy_sum = self.strategy_sum[infoset]
            strategy_sum_new = prob * strategy
            if strategy_sum.shape != strategy_sum_new.shape:
                self.strategy_sum[infoset] = strategy_sum_new
            else:
                self.strategy_sum[infoset] += strategy_sum_new

# Ejemplo mínimo de prueba (puedes agregar esto en un archivo test_cfr.py)
if __name__ == '__main__':
    class MockModel:
        def get_legal_actions(self, infoset):
            # Simula acciones legales basadas en el infoset
            if infoset == "initial":
                return ["A", "B"]
            return []

    def mock_infoset_format(state, history):
        return "initial"  # Simula un infoset inicial

    # Reemplaza la función real con la simulada para la prueba
    infoset_format = mock_infoset_format

    try:
        model = MockModel()
        trainer = CFRTrainer(model, iterations=10, max_depth=20, verbose=True, num_processes=2)
        game_state = GameState()  # Necesitas inicializar un GameState válido
        trainer.train(game_state)
        average_strategy = trainer.get_average_strategy()
        print("Entrenamiento CFR completado con éxito.")
        print("Estrategia promedio:", average_strategy)
    except CFRError as e:
        print(f"Error durante el entrenamiento CFR: {e}")