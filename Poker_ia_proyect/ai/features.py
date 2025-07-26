import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
import logging
import time
import json
from collections import Counter

try:
    from treys import Card, Evaluator
    _TREYS_AVAILABLE = True
except ImportError:
    _TREYS_AVAILABLE = False
    logging.warning("treys library not found. Hand strength evaluation will be a placeholder.")

# Configura el logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Valor por defecto para sustituir valores no numéricos
DEFAULT_NON_NUMERIC_VALUE = 0.0

# Cache interno
_FEATURE_CACHE = {}

# Funciones auxiliares para la evaluación de manos
def _evaluate_hand_strength_treys(hand: List[str], community_cards: List[str]) -> float:
    """
    Evalúa la fuerza de la mano utilizando la librería treys.

    Args:
        hand (List[str]): Lista de cartas en la mano del jugador.
        community_cards (List[str]): Lista de cartas comunitarias en el board.

    Returns:
        float: Fuerza de la mano normalizada entre 0 y 1.
    """
    try:
        if not hand or not community_cards:
            return 0.0

        # Convertir las cartas al formato de la librería treys
        hand_treys = [Card.new(card) for card in hand]
        community_cards_treys = [Card.new(card) for card in community_cards]

        # Evaluar la mano
        evaluator = Evaluator()
        rank = evaluator.evaluate(community_cards_treys, hand_treys)

        # Normalizar el rango (el rango más bajo es la mejor mano)
        return _normalize_with_nan_check(rank, 1, 7462)  # 7462 es el peor rango posible

    except Exception as e:
        logger.error(f"Error al evaluar la fuerza de la mano con treys: {e}")
        return 0.0

def _calculate_has_pair(hand: List[str], community_cards: List[str]) -> float:
    """Calcula si el jugador tiene una pareja."""
    # Combinar las cartas de la mano y las comunitarias en una sola lista
    all_cards = hand + community_cards
    
    # Extraer los rangos de todas las cartas
    ranks = [card[0] for card in all_cards]
    
    # Contar la frecuencia de cada rango
    rank_counts = Counter(ranks)
    
    # Verificar si algún rango aparece al menos dos veces (es decir, hay una pareja)
    if any(count >= 2 for count in rank_counts.values()):
        return 0.6  # Tiene al menos una pareja
    
    return 0.0

def _calculate_has_top_pair(hand: List[str], community_cards: List[str]) -> float:
    """Calcula si el jugador tiene la pareja más alta."""
    if not community_cards:
        return 0.0
    
    # Extraer los rangos de las cartas comunitarias
    community_ranks = [card[0] for card in community_cards]
    
    # Encontrar el rango más alto en las cartas comunitarias
    if not community_ranks:
        return 0.0
    max_rank = max(community_ranks)
    
    # Extraer los rangos de las cartas en la mano del jugador
    hand_ranks = [card[0] for card in hand]
    
    # Verificar si el rango más alto de las cartas comunitarias está en la mano del jugador
    if max_rank in hand_ranks:
        return 0.9  # Tiene la pareja más alta
    
    return 0.0

def _calculate_board_texture(community_cards: List[str]) -> float:
    """Evalúa si el board es seco, coordinado, etc."""
    if not community_cards:
        return 0.0
    
    # Extraer los rangos de las cartas comunitarias
    ranks = [card[0] for card in community_cards]
    
    # Verificar si todos los rangos son únicos (board seco)
    if len(set(ranks)) == len(ranks):
        return 0.2  # Board seco
    else:
        return 0.8  # Board coordinado

def _calculate_has_strong_draw(hand: List[str], community_cards: List[str]) -> float:
    """Calcula si hay un proyecto fuerte (color/escalera)."""
    # Combinar las cartas de la mano y las comunitarias en una sola lista
    all_cards = hand + community_cards
    
    # Extraer los palos de todas las cartas
    suits = [card[1] for card in all_cards]
    
    # Contar la frecuencia de cada palo
    suit_counts = Counter(suits)
    
    # Verificar si algún palo aparece al menos cuatro veces (proyecto de color fuerte)
    if any(count >= 4 for count in suit_counts.values()):
        return 1.0  # Proyecto de color fuerte
    
    # Extraer los rangos de todas las cartas
    ranks = [card[0] for card in all_cards]
    
    # Contar la frecuencia de cada rango
    rank_counts = Counter(ranks)
    
    # Verificar si algún rango aparece al menos cuatro veces (proyecto de escalera fuerte)
    if any(count >= 4 for count in rank_counts.values()):
        return 0.7  # Proyecto de escalera fuerte
    
    return 0.0

def _calculate_has_overcards(hand: List[str], community_cards: List[str], chips: float) -> float:
    """Calcula si el jugador tiene sobrecartas."""
    if not community_cards:
        return 0.0
    
    # Extraer los rangos de las cartas comunitarias
    community_ranks = [card[0] for card in community_cards]
    
    # Encontrar el rango más alto en las cartas comunitarias
    if not community_ranks:
        return 0.0
    max_rank = max(community_ranks)
    
    # Extraer los rangos de las cartas en la mano del jugador
    hand_ranks = [card[0] for card in hand]
    
    # Verificar si todos los rangos en la mano son mayores que el rango más alto en las cartas comunitarias
    if all(rank > max_rank for rank in hand_ranks):
        return 0.7  # Tiene sobrecartas
    
    return 0.0

def extract_features(game_state: Dict[str, Any], player_state: Dict[str, Any], feature_config: Optional[Dict[str, Any]] = None, verbose: bool = False, return_named: bool = False, minimal_mode: bool = False) -> Union[List[float], Dict[str, float]]:
    """
    Extrae características relevantes del estado del juego y del estado del jugador.

    Args:
        game_state (Dict[str, Any]): Diccionario que representa el estado del juego.
        player_state (Dict[str, Any]): Diccionario que representa el estado del jugador.
        feature_config (Optional[Dict[str, Any]], optional): Diccionario de configuración de características. Defaults to None.
        verbose (bool, optional): Indica si se debe mostrar información detallada. Defaults to False.
        return_named (bool, optional): Indica si se deben devolver las características con nombres. Defaults to False.
        minimal_mode (bool, optional): Indica si se debe utilizar el modo minimal. Defaults to False.

    Returns:
        Union[List[float], Dict[str, float]]: Lista de características o diccionario de características con nombres.
    """
    start_time = time.time()

    # 0. Configuración de características
    if feature_config is None:
        feature_config = {}

    include_sections = feature_config.get("include_sections", ["player", "game", "opponents", "custom"])
    clamp = feature_config.get("clamp", False)
    normalize_config = feature_config.get("normalize_config", {})
    feature_weights = feature_config.get("feature_weights", {})
    use_treys = feature_config.get("use_treys", False)
    minimal_mode_sections = feature_config.get("minimal_mode_sections", [])  # Secciones en modo minimal
    feature_ranges = feature_config.get("feature_ranges", {})  # Rangos esperados para validación

    # 1. Cache interno
    cache_key = (hash(json.dumps(game_state, sort_keys=True)), hash(json.dumps(player_state, sort_keys=True)))
    if cache_key in _FEATURE_CACHE:
        logger.debug("Usando features del cache.")
        return _FEATURE_CACHE[cache_key]

    features = []
    named_features = {}  # Para el modo explicabilidad
    feature_names = []  # Para el debug visual

    # 2. Extracción de características por sección
    if "player" in include_sections and ("player" not in minimal_mode_sections or not minimal_mode):
        player_features, player_names = _extract_player_features(player_state, game_state, feature_config, verbose, normalize_config.get("player", {}), minimal_mode)
        features.extend(player_features)
        feature_names.extend(player_names)

    if "game" in include_sections and ("game" not in minimal_mode_sections or not minimal_mode):
        game_features, game_names = _extract_game_features(game_state, player_state, feature_config, verbose, normalize_config.get("game", {}), minimal_mode)
        features.extend(game_features)
        feature_names.extend(game_names)

    if "opponents" in include_sections and ("opponents" not in minimal_mode_sections or not minimal_mode):
        opponents_features, opponents_names = _extract_opponents_features(game_state, player_state, feature_config, verbose, normalize_config.get("opponents", {}), minimal_mode)
        features.extend(opponents_features)
        feature_names.extend(opponents_names)

    if "custom" in include_sections and ("custom" not in minimal_mode_sections or not minimal_mode):
        custom_features, custom_names = _extract_custom_features(game_state, player_state, feature_config, verbose, normalize_config.get("custom", {}), minimal_mode, use_treys)
        features.extend(custom_features)
        feature_names.extend(custom_names)

    # 3. Validación de tipo y valores
    validated_features = []
    for i, feature in enumerate(features):
        feature_name = feature_names[i]
        if not isinstance(feature, (int, float)):
            logger.warning(f"Valor no numérico detectado: {feature}, sustituyendo por {DEFAULT_NON_NUMERIC_VALUE}")
            validated_feature = DEFAULT_NON_NUMERIC_VALUE
        elif np.isnan(feature) or np.isinf(feature):
            logger.warning(f"Valor NaN o Inf detectado: {feature}, sustituyendo por {DEFAULT_NON_NUMERIC_VALUE}")
            validated_feature = DEFAULT_NON_NUMERIC_VALUE
        else:
            # Detección de outliers (opcional)
            if feature_name in feature_weights:
                weight = feature_weights[feature_name]
                feature *= weight

            if clamp:
                feature = np.clip(feature, 0.0, 1.0)

            validated_feature = feature

        # Validación de rangos
        if feature_name in feature_ranges:
            min_val, max_val = feature_ranges[feature_name]
            if not min_val <= validated_feature <= max_val:
                logger.warning(f"Valor fuera del rango [{min_val}, {max_val}] detectado: {validated_feature} para la feature {feature_name}, forzando al rango.")
                validated_feature = np.clip(validated_feature, min_val, max_val)

        validated_features.append(validated_feature)

    # 4. Modo debug visual
    # if debug_visual:
    #     export_features(feature_names, validated_features, include_sections)

    # 5. Soporte para explicabilidad
    if return_named:
        named_features = dict(zip(feature_names, validated_features))
        _FEATURE_CACHE[cache_key] = named_features
        return named_features
    else:
        _FEATURE_CACHE[cache_key] = validated_features
        return validated_features

    end_time = time.time()
    logger.debug(f"Tiempo de extracción de features: {end_time - start_time:.4f} segundos")

def _extract_player_features(player_state: Dict[str, Any], game_state: Dict[str, Any], feature_config: Dict[str, Any], verbose: bool, normalize_config: Dict[str, Any], minimal_mode: bool) -> Tuple[List[float], List[str]]:
    """Extrae las características del jugador."""
    features = []
    feature_names = []
    try:
        chips = _safe_feature_extraction(player_state, 'chips', 0.0, 100.0, normalize_config.get('normalize_chips', feature_config.get('normalize_chips', True)))
        features.append(chips)
        feature_names.append("chips")
        if verbose:
            logger.info(f"Feature: chips = {chips}")

        hand_size = len(player_state.get('hand', []))
        features.append(hand_size)
        feature_names.append("hand_size")
        if verbose:
            logger.info(f"Feature: hand_size = {hand_size}")

        current_bet = game_state.get('current_bet', 0.0)
        stack_bet_diff = _normalize_with_nan_check(player_state.get('chips', 0.0) - current_bet, 0.0, 100.0)
        features.append(stack_bet_diff)
        feature_names.append("stack_bet_diff")
        if verbose:
            logger.info(f"Feature: stack_bet_diff = {stack_bet_diff}")

        is_all_in = player_state.get('is_all_in', False)
        features.append(float(is_all_in))  # Convertir a float (0.0 o 1.0)
        feature_names.append("is_all_in")
        if verbose:
            logger.info(f"Feature: is_all_in = {is_all_in}")

        player_bet_ratio = _normalize_with_nan_check(current_bet / (player_state.get('chips', 1.0) + 1e-9), 0.0, 1.0)
        features.append(player_bet_ratio)
        feature_names.append("player_bet_ratio")
        if verbose:
            logger.info(f"Feature: player_bet_ratio = {player_bet_ratio}")

    except Exception as e:
        logger.error(f"Error al extraer características del jugador: {e}")
    return features, feature_names

def _extract_game_features(game_state: Dict[str, Any], player_state: Dict[str, Any], feature_config: Dict[str, Any], verbose: bool, normalize_config: Dict[str, Any], minimal_mode: bool) -> Tuple[List[float], List[str]]:
    """Extrae las características del estado del juego."""
    features = []
    feature_names = []
    try:
        round_num = game_state.get('round_num', 0)
        pot = _safe_feature_extraction(game_state, 'pot', 0.0, 100.0, normalize_config.get('normalize_pot', feature_config.get('normalize_pot', True)))
        features.append(pot)
        feature_names.append("pot")
        if verbose:
            logger.info(f"Feature: pot = {pot}")

        community_cards_size = len(game_state.get('community_cards', []))
        features.append(community_cards_size)
        feature_names.append("community_cards_size")
        if verbose:
            logger.info(f"Feature: community_cards_size = {community_cards_size}")

        round_num_norm = _safe_feature_extraction(game_state, 'round_num', 0.0, 10.0, normalize_config.get('normalize_round_num', feature_config.get('normalize_round_num', True)))
        features.append(round_num_norm)
        feature_names.append("round_num")
        if verbose:
            logger.info(f"Feature: round_num = {round_num_norm}")

        current_bet = _safe_feature_extraction(game_state, 'current_bet', 0.0, 100.0, normalize_config.get('normalize_current_bet', feature_config.get('normalize_current_bet', True)))
        features.append(current_bet)
        feature_names.append("current_bet")
        if verbose:
            logger.info(f"Feature: current_bet = {current_bet}")

        min_raise = _safe_feature_extraction(game_state, 'min_raise', 0.0, 100.0, normalize_config.get('normalize_min_raise', feature_config.get('normalize_min_raise', True)))
        features.append(min_raise)
        feature_names.append("min_raise")
        if verbose:
            logger.info(f"Feature: min_raise = {min_raise}")

        pot_chips_ratio = _normalize_with_nan_check(game_state.get('pot', 0.0) / (player_state.get('chips', 1.0) + 1e-9), 0.0, 10.0)
        features.append(pot_chips_ratio)
        feature_names.append("pot_chips_ratio")
        if verbose:
            logger.info(f"Feature: pot_chips_ratio = {pot_chips_ratio}")

        # TODO: Implementar lógica para board_texture, has_strong_draw, has_pair, has_top_pair, has_overcards
        community_cards = game_state.get('community_cards', [])
        hand = player_state.get('hand', [])

        # Adaptar valores según la etapa del juego
        round_num = game_state.get('round_num', 0)
        if round_num == 0:  # Preflop
            board_texture = 0.0
            has_pair = 0.0
            has_top_pair = 0.0
            has_overcards = 0.0
            has_strong_draw = 0.0
        else:  # Postflop
            has_strong_draw = _calculate_has_strong_draw(hand, community_cards)
            board_texture = _calculate_board_texture(community_cards)
            has_pair = _calculate_has_pair(hand, community_cards)
            has_top_pair = _calculate_has_top_pair(hand, community_cards)
            has_overcards = _calculate_has_overcards(hand, community_cards, player_state.get('chips', 0))

        features.append(has_strong_draw)
        feature_names.append("has_strong_draw")
        features.append(board_texture)
        feature_names.append("board_texture")
        features.append(has_pair)
        feature_names.append("has_pair")
        features.append(has_top_pair)
        feature_names.append("has_top_pair")
        features.append(has_overcards)
        feature_names.append("has_overcards")

    except Exception as e:
        logger.error(f"Error al extraer características del estado del juego: {e}")
    return features, feature_names

def _extract_opponents_features(game_state: Dict[str, Any], player_state: Dict[str, Any], feature_config: Dict[str, Any], verbose: bool, normalize_config: Dict[str, Any], minimal_mode: bool) -> Tuple[List[float], List[str]]:
    """Extrae las características de los oponentes."""
    features = []
    feature_names = []
    try:
        num_active_opponents = sum(1 for opponent in game_state.get('opponents', []) if opponent.get('is_active', True))
        features.append(num_active_opponents / 5.0)  # Normalizar el número de oponentes activos
        feature_names.append("num_active_opponents")
        if verbose:
            logger.info(f"Feature: num_active_opponents = {num_active_opponents / 5.0}")
    except Exception as e:
        logger.error(f"Error al extraer características de los oponentes: {e}")
    return features, feature_names

def _extract_custom_features(game_state: Dict[str, Any], player_state: Dict[str, Any], feature_config: Dict[str, Any], verbose: bool, normalize_config: Dict[str, Any], minimal_mode: bool, use_treys: bool) -> Tuple[List[float], List[str]]:
    """Extrae las características personalizadas."""
    features = []
    feature_names = []
    try:
        aggression_factor = _calculate_aggression_factor(player_state)
        features.append(aggression_factor)
        feature_names.append("aggression_factor")
        if verbose:
            logger.info(f"Feature: aggression_factor = {aggression_factor}")

        # Adaptar hand_strength según la etapa del juego
        round_num = game_state.get('round_num', 0)
        if round_num == 0:  # Preflop
            hand_strength = 0.5  # Valor por defecto preflop
        else:
            hand_strength = _evaluate_hand_strength(player_state.get('hand', []), game_state.get('community_cards', []), use_treys)

        features.append(hand_strength)
        feature_names.append("hand_strength")
        if verbose:
            logger.info(f"Feature: hand_strength = {hand_strength}")

        custom_features_config = feature_config.get('custom_features', {})
        if isinstance(custom_features_config, dict):
            for feature_name, feature_func in custom_features_config.items():
                if isinstance(feature_func, dict) and 'func' in feature_func and callable(feature_func['func']):
                    func = feature_func['func']
                    validate = feature_func.get('validate', True)  # Habilitado por defecto
                    if validate:
                        try:
                            custom_feature = func(game_state, player_state)
                            features.append(custom_feature)
                            feature_names.append(feature_name)
                            if verbose:
                                logger.info(f"Feature: {feature_name} = {custom_feature}")
                        except Exception as e:
                            logger.error(f"Error al calcular la característica personalizada '{feature_name}': {e}")
                    else:
                        logger.warning(f"La característica personalizada '{feature_name}' está desactivada debido a validación fallida.")
                else:
                    logger.warning(f"La característica personalizada '{feature_name}' no es una función callable o no tiene el formato correcto.")
    except Exception as e:
        logger.error(f"Error al procesar características personalizadas: {e}")
    return features, feature_names

def _safe_feature_extraction(data: Dict[str, Any], key: str, default_value: float, max_value: float, normalize: bool) -> float:
    """
    Extrae una característica de forma segura, manejando valores faltantes y normalizando si es necesario.
    """
    try:
        value = data.get(key, default_value)
        if normalize:
            return _normalize_with_nan_check(value, 0.0, max_value)
        return float(value)
    except Exception as e:
        logger.error(f"Error al extraer la característica '{key}': {e}")
        return default_value

def _calculate_aggression_factor(player_state: Dict[str, Any]) -> float:
    """
    Calcula un factor de agresión basado en el historial de acciones del jugador.
    """
    try:
        raises = player_state.get('raises', 0)
        bets = player_state.get('bets', 0)
        calls = player_state.get('calls', 0)
        return (raises + bets) / (calls + 1)
    except Exception as e:
        logger.error(f"Error al calcular el factor de agresión: {e}")
        return 0.0

def _evaluate_hand_strength(hand: List[str], community_cards: List[str], use_treys: bool = False) -> float:
    """
    Evalúa la fuerza de la mano del jugador utilizando una librería de evaluación de manos.
    """
    if use_treys and _TREYS_AVAILABLE:
        return _evaluate_hand_strength_treys(hand, community_cards)
    else:
        return 0.5

def _normalize_with_nan_check(value: float, min_value: float, max_value: float) -> float:
    """
    Normaliza un valor al rango [0, 1], manejando NaN, Inf y valores extremos.
    """
    try:
        if max_value == min_value:
            return 0.0  # Evitar división por cero
        normalized_value = (value - min_value) / (max_value - min_value)
        if np.isnan(normalized_value) or np.isinf(normalized_value):
            logger.warning(f"Valor normalizado NaN o Inf detectado: {value}, devolviendo {DEFAULT_NON_NUMERIC_VALUE}")
            return DEFAULT_NON_NUMERIC_VALUE
        return np.clip(normalized_value, 0.0, 1.0)  # Asegurar el rango [0, 1]
    except Exception as e:
        logger.error(f"Error al normalizar el valor: {e}")
        return DEFAULT_NON_NUMERIC_VALUE
    
def one_hot_encode_cards(cards: List[str]) -> np.ndarray:
    """
    Codifica una lista de cartas como un vector one-hot de 52 posiciones.
    """
    ranks = "23456789TJQKA"
    suits = "shdc"
    card_to_idx = {r + s: i for i, (r, s) in enumerate((r, s) for r in ranks for s in suits)}
    vec = np.zeros(52)
    for card in cards:
        card = card.lower()
        if len(card) == 2 and card in card_to_idx:
            idx = card_to_idx[card]
            vec[idx] = 1
    return vec

def extract_card_features(hand: List[str], community_cards: List[str]) -> np.ndarray:
    """
    Extrae features one-hot para la mano y las comunitarias.
    """
    hand_vec = one_hot_encode_cards(hand)
    board_vec = one_hot_encode_cards(community_cards)
    return np.concatenate([hand_vec, board_vec])

def extract_action_history_features(history: List[Any], max_actions: int = 10) -> np.ndarray:
    """
    Codifica las últimas acciones como un vector one-hot plano.
    """
    action_types = [ACTION_BET, ACTION_CALL, ACTION_CHECK, ACTION_FOLD, ACTION_ALL_IN, ACTION_RAISE]
    action_to_idx = {a: i for i, a in enumerate(action_types)}
    vec = np.zeros((max_actions, len(action_types)))
    for i, entry in enumerate(history[-max_actions:]):
        if isinstance(entry, (tuple, list)) and len(entry) >= 2:
            action = entry[1]
            idx = action_to_idx.get(action, None)
            if idx is not None:
                vec[i, idx] = 1
    return vec.flatten()

def extract_position_feature(game_state: Dict[str, Any], player_state: Dict[str, Any]) -> float:
    """
    Devuelve 1.0 si el jugador es dealer, 0.0 si es BB.
    """
    roles = game_state.get("roles", {})
    player_name = player_state.get("name", "")
    return 1.0 if roles.get(player_name, "") == "dealer" else 0.0

def extract_stack_relative_feature(player_state: Dict[str, Any], game_state: Dict[str, Any]) -> float:
    """
    Devuelve el stack del jugador normalizado por el pot.
    """
    pot = game_state.get("pot", 1.0)
    chips = player_state.get("chips", 0.0)
    return chips / (pot + 1e-9)

def extract_full_feature_vector(game_state: Dict[str, Any], player_state: Dict[str, Any], history: List[Any]) -> np.ndarray:
    """
    Extrae un vector de features completo, combinando todas las mejoras.
    """
    # One-hot de cartas
    cards_feat = extract_card_features(player_state.get("hand", []), game_state.get("community_cards", []))
    # Historial de acciones
    history_feat = extract_action_history_features(history)
    # Posición
    pos_feat = np.array([extract_position_feature(game_state, player_state)])
    # Stack relativo
    stack_feat = np.array([extract_stack_relative_feature(player_state, game_state)])
    # Otros features numéricos (puedes agregar más)
    pot_feat = np.array([_normalize_with_nan_check(game_state.get("pot", 0.0), 0.0, 100.0)])
    # Concatenar todo
    return np.concatenate([cards_feat, history_feat, pos_feat, stack_feat, pot_feat])

