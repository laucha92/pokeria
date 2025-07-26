import torch
import torch.nn as nn
import logging
import numpy as np
from typing import Optional, List, Union, Tuple, Dict, Callable, Any
from ai.mcts import MCTS_Agent
from exceptions.ai_exceptions import PolicyActionSelectionError
import random
import json
import uuid
from collections import deque, OrderedDict
import os
from scipy.stats import entropy  # Para la entropía
from scipy.special import rel_entr  # Para la divergencia KL
import traceback  # Para loguear stack traces
import unittest  # Para pruebas unitarias
from sklearn.decomposition import PCA  # Para PCA

from models.simple_weight_model import SimpleWeightModel
from policy.fallback_policies import UniformFallbackPolicy, RandomFallbackPolicy, HeuristicFallbackPolicy
from analysis.game_context_analyzer import GameContextAnalyzer
from utils.model_io import save_model, load_model

# Configura el logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Policy:
    """
    Clase que implementa una política híbrida avanzada para seleccionar acciones en un juego.
    """
    def __init__(
        self,
        model: nn.Module,
        mcts_agent: MCTS_Agent,
        cfr_data: Optional[dict] = None,
        strategy_weights: Optional[Dict[str, float]] = None,
        selection_mode: str = 'argmax',
        epsilon: float = 0.1,
        temperature: float = 1.0,
        debug_mode: bool = False,
        adaptive_weights: bool = False,
        state_size: int = 10,
        game_context_analyzer: Optional[GameContextAnalyzer] = None,
        use_confidence: bool = False,
        initial_temperature: float = 1.0,
        temperature_decay: float = 0.995,
        min_temperature: float = 0.1,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        fallback_policy: Optional[FallbackPolicy] = None,
        external_logging_callback: Optional[Callable] = None,
        replay_mode: bool = False,
        use_cache: bool = False,
        cache_size: int = 100,
        model_save_path: str = "policy_weights.pth",
        initial_confidence_threshold: float = 0.1,
        confidence_threshold_decay: float = 0.999,
        min_confidence_threshold: float = 0.01,
        pca_components: int = 3,  # Número de componentes PCA a utilizar
        heuristic_fallback_policy: Optional[HeuristicFallbackPolicy] = None,
        use_weighted_voting: bool = True,
        trust_decay_factor: float = 0.999,
        min_trust: float = 0.01,
        cache_invalidation_threshold: float = 0.1,
        training_mode: bool = False,
        exploration_factor: float = 0.1
    ):
        self.model = model
        self.mcts_agent = mcts_agent
        self.cfr_data = cfr_data
        self.device = next(model.parameters()).device
        self.strategy_weights = strategy_weights if strategy_weights is not None else {"model": 0.5, "mcts": 0.5}
        self.selection_mode = selection_mode
        self.epsilon = epsilon
        self.temperature = temperature
        self.debug_mode = debug_mode
        self.cfr_strategy = None
        self.adaptive_weights = adaptive_weights
        self.state_size = state_size
        self.game_context_analyzer = game_context_analyzer
        self.use_confidence = use_confidence
        self.decision_history = []
        self.callbacks = {}
        self.initial_temperature = initial_temperature
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.current_temperature = initial_temperature
        self.current_epsilon = epsilon
        self.fallback_policy = fallback_policy if fallback_policy is not None else UniformFallbackPolicy()
        self.heuristic_fallback_policy = heuristic_fallback_policy
        self.external_logging_callback = external_logging_callback
        self.replay_mode = replay_mode
        self.use_cache = use_cache
        # Usar OrderedDict para implementar LRU
        self.cache: OrderedDict[Tuple[float, ...], int] = OrderedDict()
        self.cache_size = cache_size
        self.model_save_path = model_save_path
        self.rolling_rewards = {name: deque(maxlen=10) for name in self.strategy_weights}
        self.strategy_performance = {name: 0.0 for name in self.strategy_weights}
        self.num_strategies = len(self.strategy_weights)
        #self.confidence_threshold_adjuster = ConfidenceThresholdAdjuster(initial_confidence_threshold, confidence_threshold_decay, min_confidence_threshold)
        #self.strategy_conflict_detector = StrategyConflictDetector(pca_components)
        if adaptive_weights:
            self.weight_model = SimpleWeightModel(self.num_strategies, self.state_size).to(self.device)
            self.optimizer = torch.optim.Adam(self.weight_model.parameters(), lr=0.001)
        self.use_weighted_voting = use_weighted_voting
        self.trust = {name: 1.0 for name in self.strategy_weights}
        self.trust_decay_factor = trust_decay_factor
        self.min_trust = min_trust
        self.cache_invalidation_threshold = cache_invalidation_threshold
        self.training_mode = training_mode
        self.exploration_factor = exploration_factor
        self.self_diagnostics()

    def register_error_callback(self, callback: Callable):
        """
        Registra un callback para eventos de error críticos.
        """
        self.error_callbacks.append(callback)

    def _trigger_error_callbacks(self, error: Exception, context: dict):
        for cb in self.error_callbacks:
            try:
                cb(error, context)
            except Exception as e:
                logger.error(f"Error en error_callback: {e}")

    def explain_last_decision(self) -> Dict[str, Any]:
        """
        Devuelve un resumen explicativo de la última decisión tomada.
        """
        return self.last_profile.copy() if self.last_profile else {}

    def set_multi_player_mode(self, enabled: bool = True):
        """
        Activa el modo multi-jugador (hook para futuras extensiones).
        """
        self.multi_player_mode = enabled    

    def load_cfr_strategy(self) -> None:
        """
        Carga la estrategia CFR desde los datos CFR.
        """
        if self.cfr_data and 'strategy' in self.cfr_data:
            self.cfr_strategy = self.cfr_data['strategy']
            logger.info("Estrategia CFR cargada correctamente.")
        else:
            logger.warning("No se encontraron datos CFR válidos. CFR strategy no será utilizada.")

    def _validate_and_clip_strategy(self, strategy: np.ndarray) -> np.ndarray:
        """
        Valida y recorta una estrategia para asegurar que esté en el rango [0, 1], sume 1 y no contenga NaNs.
        """
        if not isinstance(strategy, np.ndarray):
            raise ValueError("La estrategia debe ser un numpy array.")
        if len(strategy.shape) != 1:
            raise ValueError("La estrategia debe ser un array unidimensional.")
        strategy = np.nan_to_num(strategy, nan=0.0)
        strategy = np.clip(strategy, 0, 1)
        if np.sum(strategy) == 0:
            strategy = self.fallback_policy.get_fallback_strategy(len(strategy))
        else:
            strategy /= np.sum(strategy)
        return strategy

    def get_strategy_confidence(self, strategy_name: str, strategy_output: Any) -> float:
        """
        Calcula la confianza de una estrategia.
        """
        if strategy_name == "model":
            if not isinstance(strategy_output, torch.Tensor):
                raise TypeError("strategy_output debe ser un torch.Tensor para la estrategia 'model'")
            probabilities = torch.softmax(strategy_output, dim=-1)
            entropy_val = entropy(probabilities.cpu().detach().numpy()[0])
            confidence = 1.0 - (entropy_val / np.log(probabilities.size(-1))).item()
            return confidence
        elif strategy_name == "mcts":
            if not isinstance(self.mcts_agent.num_simulations, int):
                raise TypeError("mcts_agent.num_simulations debe ser un entero para la estrategia 'mcts'")
            confidence = min(1.0, self.mcts_agent.num_simulations / 1000)
            return confidence
        elif strategy_name == "cfr":
            return 0.7
        else:
            return 0.0

    def select_action(self, game_state: List[float], round_state: Dict) -> int:
        """
        Selecciona una acción basada en la política híbrida avanzada.
        """
        import time
        t0 = time.time()
        try:
            if not isinstance(game_state, list):
                raise ValueError("El game_state debe ser una lista.")
            if not all(isinstance(x, (int, float)) for x in game_state):
                raise ValueError("El game_state debe contener solo números.")

            game_state_tensor = torch.tensor(game_state, dtype=torch.float).to(self.device)

            trace_id = str(uuid.uuid4())
            # 1. Cache
            if self.use_cache:
                action = self._get_cached_action(tuple(game_state))
                if action is not None:
                    logger.debug(f"Usando estrategia en cache para game_state: {game_state}")
                    return action

            strategies: Dict[str, np.ndarray] = {}
            raw_strategies: Dict[str, Any] = {}
            confidences: Dict[str, float] = {}

            self._trigger_callback("before_select_action", game_state=game_state)

            # 2. Obtener la estrategia del modelo
            try:
                self.model.eval()
                with torch.no_grad():
                    output = self.model(game_state_tensor.unsqueeze(0))
                    raw_strategies["model"] = output
                    model_strategy: np.ndarray = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
                    model_strategy = self._validate_and_clip_strategy(model_strategy)
                    strategies["model"] = model_strategy
                    confidences["model"] = self.get_strategy_confidence("model", output)
            except Exception as e:
                logger.warning(f"Error al obtener la estrategia del modelo: {e}. Usando fallback. {traceback.format_exc()}")
                strategies["model"] = self.fallback_policy.get_fallback_strategy(len(game_state))
                confidences["model"] = 0.0

            # 3. Obtener la estrategia del agente MCTS
            try:
                mcts_strategy: np.ndarray = self.mcts_agent.get_action_probs(game_state)
                raw_strategies["mcts"] = mcts_strategy
                mcts_strategy = self._validate_and_clip_strategy(mcts_strategy)
                strategies["mcts"] = mcts_strategy
                confidences["mcts"] = self.get_strategy_confidence("mcts", self.mcts_agent.num_simulations)
            except Exception as e:
                logger.warning(f"Error al obtener la estrategia del MCTS: {e}. Usando fallback. {traceback.format_exc()}")
                strategies["mcts"] = self.fallback_policy.get_fallback_strategy(len(game_state))
                confidences["mcts"] = 0.0

            # 4. Obtener la estrategia CFR si está disponible
            if self.cfr_strategy is not None:
                try:
                    cfr_strategy: np.ndarray = self._validate_and_clip_strategy(np.array(self.cfr_strategy))
                    strategies["cfr"] = cfr_strategy
                    confidences["cfr"] = self.get_strategy_confidence("cfr", None)
                except Exception as e:
                    logger.warning(f"Error al obtener la estrategia CFR: {e}. Ignorando CFR strategy. {traceback.format_exc()}")
                    strategies["cfr"] = self.fallback_policy.get_fallback_strategy(len(game_state))
                    confidences["cfr"] = 0.0

            # 5. Blending dinámico autoajustable por contexto real
            weights = self._calculate_weights(game_state, strategies, confidences, round_state)

            # 6. Sistema de confianza y ranking más sofisticado
            ranked_strategies = self._voting_ranking(strategies, confidences, weights)

            # 7. Combinar las estrategias
            blended_strategy = self.blend_strategies(ranked_strategies, weights)

            # 8. Exploración estratégica
            if self.training_mode:
                blended_strategy = self._explore_strategy(blended_strategy)

            # 9. Seleccionar la acción basada en el modo de selección
            action = self._select_action_from_strategy(blended_strategy)

            # 10. Guardar la decisión en el historial
            if not self.replay_mode:
                decision_data = {
                    "trace_id": trace_id,
                    "game_state": game_state,
                    "strategies": {k: v.tolist() for k, v in strategies.items()},
                    "raw_strategies": {k: str(v) for k, v in raw_strategies.items()},  # Convertir a string para serialización
                    "weights": weights,
                    "blended_strategy": blended_strategy.tolist(),
                    "action": int(action),
                    "temperature": self.current_temperature,
                    "epsilon": self.current_epsilon,
                    "confidences": confidences
                }
                self.decision_history.append(decision_data)

            # 11. Logueo estructurado
            log_data = {
                "trace_id": trace_id,
                "game_state": game_state,
                "strategies": {k: v.tolist() for k, v in strategies.items()},
                "weights": weights,
                "blended_strategy": blended_strategy.tolist(),
                "action": int(action),
                "round": round_state.get("round", "PREFLOP"),
                "role": round_state.get("role", "NA")
            }
            logger.info(json.dumps(log_data))

            # 12. Callback externo
            if self.external_logging_callback:
                self.external_logging_callback(decision_data)

            if self.debug_mode:
                logger.debug(f"Estrategias: {strategies}")
                logger.debug(f"Pesos: {weights}")
                logger.debug(f"Estrategia combinada: {blended_strategy}")
                logger.debug(f"Acción seleccionada: {action}")

            logger.info(f"Acción seleccionada: {action}")

            self._trigger_callback("after_select_action", game_state=game_state, action=action, strategies=strategies, weights=weights, blended_strategy=blended_strategy)

            # 13. Decaimiento de la temperatura y epsilon
            self._decay_temperature()
            self._decay_epsilon()

            # 14. Ajuste dinámico del umbral de confianza
            #self.confidence_threshold_adjuster.adjust_threshold(strategies, action, confidences)

            # 15. Manejo y exploración con cache mejorado
            if self.use_cache:
                self._update_cache(tuple(game_state), int(action), round_state)
            
            self.last_profile = {
                "action": int(action),
                "blended_strategy": blended_strategy.tolist(),
                "weights": weights,
                "confidences": confidences,
                "strategies": {k: v.tolist() for k, v in strategies.items()},
                "round_state": round_state,
                "trace_id": trace_id
            }
            if self.debug_mode:
                logger.info(f"[PROFILE] Tiempo de select_action: {time.time() - t0:.4f}s")
            return int(action)
        except Exception as e:
            logger.error(f"Error al seleccionar la acción: {e} {traceback.format_exc()}")
            self._trigger_error_callbacks(e, {"game_state": game_state, "round_state": round_state})
            # Fallback robusto: siempre devuelve una acción válida
            try:
                fallback = self.fallback_policy.get_fallback_strategy(len(game_state))
                return int(np.argmax(fallback))
            except Exception as fallback_e:
                logger.critical(f"Fallback también falló: {fallback_e}")
                raise PolicyActionSelectionError("No se pudo seleccionar la acción ni con fallback") from e

    def _calculate_weights(self, game_state: List[float], strategies: Dict[str, np.ndarray], confidences: Dict[str, float], round_state: Dict) -> Dict[str, float]:
        """
        Calcula los pesos para cada estrategia basándose en el contexto del juego y la confianza.
        """
        if self.game_context_analyzer:
            context_weights = self.game_context_analyzer.analyze_context(game_state, round_state)
        else:
            context_weights = self.strategy_weights

        if self.use_confidence:
            for strategy_name in strategies:
                context_weights[strategy_name] *= confidences[strategy_name]

        # Incorporar trust decay
        for strategy_name in strategies:
            context_weights[strategy_name] *= self.trust[strategy_name]
            self.trust[strategy_name] = max(self.min_trust, self.trust[strategy_name] * self.trust_decay_factor)

        total_weight = sum(context_weights.values())
        if total_weight == 0:
            weights = {k: 1.0 / len(strategies) for k in strategies}
        else:
            weights = {k: v / total_weight for k, v in context_weights.items()}

        return weights

    def _voting_ranking(self, strategies: Dict[str, np.ndarray], confidences: Dict[str, float], weights: Dict[str, float]) -> Dict[str, np.ndarray]:
        """
        Implementa un mecanismo de voting/ranking entre estrategias.
        """
        if self.use_weighted_voting:
            # Voto ponderado
            ranked_strategies = {name: strategy * weights[name] for name, strategy in strategies.items()}
        else:
            # Corte por umbral
            filtered_strategies = {name: strategy for name, strategy in strategies.items() if confidences[name] >= self.confidence_threshold_adjuster.current_confidence_threshold}
            ranked_strategies = dict(sorted(filtered_strategies.items(), key=lambda item: confidences[item[0]], reverse=True))

        return ranked_strategies

    def blend_strategies(self, strategies: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
        """
        Mezcla diferentes estrategias utilizando pesos especificados.
        """
        # Validar que todas las estrategias tengan la misma longitud
        strategy_lengths = {name: len(strategy) for name, strategy in strategies.items()}
        if len(set(strategy_lengths.values())) > 1:
            raise ValueError(f"Las estrategias tienen longitudes diferentes: {strategy_lengths}")

        # Inicializar la estrategia combinada con ceros
        blended_strategy = np.zeros_like(next(iter(strategies.values())))

        for strategy_name, strategy in strategies.items():
            weight = weights.get(strategy_name, 0.0)
            blended_strategy += weight * strategy

        return blended_strategy

    def _explore_strategy(self, blended_strategy: np.ndarray) -> np.ndarray:
        """
        Implementa blending dirigido a la exploración durante el entrenamiento.
        """
        # Forzar sampling más variado según entropía
        entropy_value = entropy(blended_strategy)
        exploration_boost = self.exploration_factor * (1.0 - (entropy_value / np.log(len(blended_strategy))))
        # Agregar ruido a la estrategia combinada
        noise = np.random.normal(0, exploration_boost, len(blended_strategy))
        blended_strategy += noise
        blended_strategy = np.clip(blended_strategy, 0, 1)
        blended_strategy /= np.sum(blended_strategy)
        return blended_strategy

    def _select_action_from_strategy(self, blended_strategy: np.ndarray) -> int:
        """
        Selecciona una acción basada en el modo de selección.
        """
        if self.selection_mode == 'argmax':
            action = np.argmax(blended_strategy)
        elif self.selection_mode == 'sampling':
            action = self._sample_action(blended_strategy, temperature=self.current_temperature)
        elif self.selection_mode == 'epsilon_greedy':
            if random.random() < self.current_epsilon:
                action = random.randint(0, len(blended_strategy) - 1)
            else:
                action = np.argmax(blended_strategy)
        else:
            raise ValueError(f"Modo de selección no válido: {self.selection_mode}")
        return int(action)

    def _sample_action(self, strategy: np.ndarray, temperature: float = 1.0) -> int:
        """
        Muestrea una acción de una distribución de probabilidad utilizando softmax con temperatura.
        """
        probabilities = np.exp(np.log(strategy) / temperature)
        probabilities /= np.sum(probabilities)
        return int(np.random.choice(len(strategy), p=probabilities))

    def update_weights(self, game_state: List[float], action: int, reward: float):
        """
        Actualiza los pesos del modelo basándose en la recompensa obtenida después de cada acción.
        """
        if self.adaptive_weights:
            try:
                self.optimizer.zero_grad()
                state_tensor = torch.tensor(game_state, dtype=torch.float).to(self.device)
                predicted_weights = self.weight_model(state_tensor)

                # Crear etiquetas basadas en el rendimiento de las estrategias
                target_weights = torch.tensor([self.strategy_performance[name] for name in self.strategy_weights], dtype=torch.float).to(self.device)
                # Usar una función de pérdida más sofisticada (ejemplo: CrossEntropyLoss)
                loss = nn.functional.cross_entropy(predicted_weights.unsqueeze(0), target_weights.unsqueeze(0))
                loss.backward()
                self.optimizer.step()
            except Exception as e:
                logger.error(f"Error al actualizar los pesos: {e} {traceback.format_exc()}")

    def _decay_temperature(self):
        """
        Reduce la temperatura para el muestreo de acciones.
        """
        self.current_temperature = max(self.min_temperature, self.current_temperature * self.temperature_decay)

    def _decay_epsilon(self):
        """
        Reduce el valor de epsilon para epsilon-greedy.
        """
        self.current_epsilon = max(self.min_epsilon, self.current_epsilon * self.epsilon_decay)

    def add_callback(self, event: str, callback: Callable):
        """
        Agrega un callback para un evento específico.
        """
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)

    def _trigger_callback(self, event: str, **kwargs: Any):
        """
        Ejecuta los callbacks para un evento específico.
        """
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                callback(**kwargs)

    def _detect_strategy_conflict(self, strategies: Dict[str, np.ndarray]) -> bool:
        """
        Detecta conflictos entre estrategias utilizando la divergencia de Kullback-Leibler (KL),
        la correlación y el análisis de componentes principales (PCA).
        """
        if len(strategies) < 2:
            return False

        strategy_list = list(strategies.values())
        num_actions = len(strategy_list[0])

        # Agregar un valor pequeño para evitar divisiones por cero
        epsilon = 1e-6
        strategy1 = strategy_list[0] + epsilon
        strategy2 = strategy_list[1] + epsilon

        # Normalizar las estrategias
        strategy1 /= np.sum(strategy1)
        strategy2 /= np.sum(strategy2)

        # Calcular la divergencia de Kullback-Leibler
        kl_divergence = np.sum(strategy1 * np.log(strategy1 / strategy2))

        # Calcular la correlación entre las estrategias
        correlation = np.corrcoef(strategy1, strategy2)[0, 1]

        # Análisis de Componentes Principales (PCA)
        try:
            pca = PCA(n_components=self.pca_components)
            pca.fit(np.array([strategy1, strategy2]))
            explained_variance_ratio = pca.explained_variance_ratio_
            # Si la varianza explicada por los primeros componentes es baja, podría indicar un conflicto
            pca_conflict = np.sum(explained_variance_ratio) < 0.5
        except Exception as e:
            logger.warning(f"Error al realizar PCA: {e} {traceback.format_exc()}")
            pca_conflict = False

        # Definir umbrales para detectar conflictos
        kl_threshold = 0.5
        correlation_threshold = 0.5

        # Detectar conflicto si la divergencia KL es alta o la correlación es baja
        return kl_divergence > kl_threshold or correlation < correlation_threshold or pca_conflict

    def self_diagnostics(self):
        """
        Realiza una auto-evaluación de consistencia.
        """
        logger.info("Realizando auto-evaluación de consistencia...")

        # 1. Verificar cobertura de estrategias
        if not self.strategy_weights:
            logger.warning("Advertencia: No se han definido pesos estratégicos.")

        # 2. Verificar integridad de pesos
        total_weight = sum(self.strategy_weights.values())
        if not np.isclose(total_weight, 1.0):
            logger.warning(f"Advertencia: Los pesos estratégicos no suman 1.0 (suma: {total_weight}).")

        # 3. Verificar dimensiones (Placeholder)
        # Implementar lógica para verificar dimensiones aquí

        # 4. Verificar calidad de blending
        self._evaluate_blending_quality(self.strategy_weights, self.blend_strategies)

        # 5. Detectar conflictos sutiles entre estrategias
        self._detect_subtle_conflicts()

        logger.info("Auto-evaluación de consistencia completada.")

    def _evaluate_blending_quality(self, strategies: Dict[str, float], blended_strategy: np.ndarray):
        """
        Evalúa la calidad del blending utilizando métricas cuantitativas.
        """
        kl_divergences = {}
        entropies = {}

        # Calcular la entropía de cada estrategia
        for name, strategy_weight in strategies.items():
            strategy = np.array([strategy_weight])  # Convertir el peso en un array de NumPy
            entropies[name] = entropy(strategy)

        # Calcular la divergencia KL entre cada estrategia y la estrategia combinada
        for name, strategy_weight in strategies.items():
            strategy = np.array([strategy_weight])  # Convertir el peso en un array de NumPy
            kl_divergences[name] = rel_entr(strategy, blended_strategy).sum()

        logger.info(f"Entropías de las estrategias: {entropies}")
        logger.info(f"Divergencias KL de las estrategias: {kl_divergences}")

    def _detect_subtle_conflicts(self):
        """
        Detecta conflictos más sutiles entre estrategias (confusión mutua, predicciones dominadas).
        """
        # Lógica para detectar conflictos sutiles
        pass

    def save_model(self, filepath: str):
        """
        Guarda el modelo de pesos y el optimizador.
        """
        torch.save({
            'model_state_dict': self.weight_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        logger.info(f"Modelo guardado en {filepath}")

    def load_model(self, filepath: str):
        """
        Carga el modelo de pesos y el optimizador.
        """
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.weight_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Modelo cargado desde {filepath}")
        else:
            logger.warning(f"No se encontró el modelo en {filepath}")

    def _update_cache(self, game_state_tuple: Tuple[float, ...], action: int, round_state: Dict) -> None:
        """
        Actualiza el cache con la nueva acción.
        """
        if game_state_tuple in self.cache:
            # Invalidación temprana de cache
            if self._should_invalidate_cache(game_state_tuple, round_state):
                del self.cache[game_state_tuple]
                logger.debug(f"Invalidando entrada de cache para game_state: {game_state_tuple}")
                return
            self.cache.move_to_end(game_state_tuple)
        self.cache[game_state_tuple] = action
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)  # Eliminar el elemento menos recientemente usado (LRU)

    def _should_invalidate_cache(self, game_state_tuple: Tuple[float, ...], round_state: Dict) -> bool:
        """
        Determina si se debe invalidar la entrada de cache.
        """
        # Lógica para determinar si se debe invalidar la entrada de cache
        # Ejemplo: Invalidar si el tamaño del bote ha cambiado significativamente
        if 'pot_size' in round_state:
            cached_pot_size = self.cache.get(game_state_tuple, {}).get('pot_size')
            if cached_pot_size is not None:
                pot_change = abs(round_state['pot_size'] - cached_pot_size) / cached_pot_size
                if pot_change > self.cache_invalidation_threshold:
                    return True
        return False