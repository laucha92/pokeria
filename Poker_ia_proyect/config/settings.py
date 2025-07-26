import json
import logging
import os
from typing import Dict, Union, Tuple, Any, Optional
import argparse  # Importar argparse

from exceptions import config_exceptions
from utils.logger import setup_logger  # Importar setup_logger

# Inicializar el logger (esto se hará solo una vez)
log = setup_logger("config_loader", "logs/config.log")

# Configuración por defecto
DEFAULT_CONFIG = {
    "player1_name": "AI Player",
    "player2_name": "Human Player",
    "initial_chips": 1000,
    "small_blind": 10,
    "big_blind": 20,
    "log_file": "logs/game.log",
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "seed": None,  # Para reproducibilidad, establecer a un entero
    "use_vision": False,
    "camera_id": 0,
    "camera_params_path": "calibration/camera_params.json",
    "card_detection_model_path": "models/card_detection_model.pth",
    "chip_detection_model_path": "models/chip_detection_model.pth",
    "ai_model_path": "models/poker_net.pth",
    "ai_input_size": 10,
    "ai_hidden_size": 20,
    "ai_output_size": 3,
    "action_selection_strategy": "epsilon_greedy",  # "epsilon_greedy" o "softmax"
    "epsilon": 0.1,  # Tasa de exploración para epsilon-greedy
    "temperature": 1.0,  # Temperatura para softmax
    "player2_is_ai": False, #Indica si el jugador 2 es la IA o no
    "max_rounds": 1000, #Máximo número de rondas
    "stats_interval": 10, #Intervalo para mostrar estadísticas
    "game_history_filename": "game_history", #Nombre del archivo para guardar el historial del juego
    "game_history_format": "json" #Formato para guardar el historial del juego
}

class Config:
    """
    Clase para manejar la configuración del juego.

    Encapsula la configuración, proporciona validación y permite la recarga.
    """
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Inicializa la configuración con un diccionario.

        Args:
            config_dict (Dict[str, Any]): Diccionario con los parámetros de configuración.
        """
        self._config: Dict[str, Any] = DEFAULT_CONFIG.copy()
        self._config.update(config_dict)
        self.validate_config()
        self.apply_environment_variables()

    def __getitem__(self, key: str) -> Any:
        """
        Permite acceder a los valores de configuración como si fuera un diccionario.

        Args:
            key (str): La clave del valor a obtener.

        Returns:
            Any: El valor de configuración.
        """
        return self._config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Permite modificar los valores de configuración como si fuera un diccionario.

        Args:
            key (str): La clave del valor a modificar.
            value (Any): El nuevo valor para la clave.
        """
        self._config[key] = value
        self.validate_config()  # Revalida la configuración después de modificarla

    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtiene un valor de configuración con un valor por defecto si no existe.

        Args:
            key (str): La clave del valor a obtener.
            default (Any, optional): Valor por defecto si la clave no existe. Defaults to None.

        Returns:
            Any: El valor de configuración o el valor por defecto.
        """
        return self._config.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        """
        Devuelve una copia del diccionario de configuración.

        Returns:
            Dict[str, Any]: Una copia del diccionario de configuración.
        """
        return self._config.copy()

    def reload(self, config_dict: Dict[str, Any]) -> None:
        """
        Recarga la configuración con un nuevo diccionario.

        Args:
            config_dict (Dict[str, Any]): Nuevo diccionario de configuración.
        """
        log.info("Recargando configuración...")
        self._config = DEFAULT_CONFIG.copy()
        self._config.update(config_dict)
        self.validate_config()
        log.info("Configuración recargada.")

    def validate_config(self) -> None:
        """
        Valida la configuración para asegurar que los valores sean del tipo correcto.

        Raises:
            ValueError: Si algún valor no es del tipo esperado.
        """
        expected_types: Dict[str, Union[type, Tuple[type, type]]] = {
            "player1_name": str,
            "player2_name": str,
            "initial_chips": int,
            "small_blind": int,
            "big_blind": int,
            "log_file": str,
            "log_level": str,
            "seed": (int, type(None)),  # Puede ser int o None
            "use_vision": bool,
            "camera_id": int,
            "camera_params_path": str,
            "card_detection_model_path": str,
            "chip_detection_model_path": str,
            "ai_model_path": str,
            "ai_input_size": int,
            "ai_hidden_size": int,
            "ai_output_size": int,
            "action_selection_strategy": str,  # "epsilon_greedy" o "softmax"
            "epsilon": float,  # Tasa de exploración para epsilon-greedy
            "temperature": float,  # Temperatura para softmax
            "player2_is_ai": bool,
            "max_rounds": int,
            "stats_interval": int,
            "game_history_filename": str,
            "game_history_format": str
        }

        for key in list(self._config.keys()):
            if key not in expected_types:
                log.warning(f"Clave extra '{key}' encontrada en la configuración. Ignorando.")
                del self._config[key]  # Ignora claves extra

        for key, expected_type in expected_types.items():
            value = self._config.get(key)
            if value is None:
                log.warning(f"La clave '{key}' no está presente en la configuración, usando el valor por defecto.")
                self._config[key] = DEFAULT_CONFIG[key]  # Completa con el valor por defecto
                log.info(f"Aplicando valor por defecto '{DEFAULT_CONFIG[key]}' para la clave '{key}'.")
                continue

            # Permite una tupla de tipos para la validación
            if isinstance(expected_type, tuple):
                if not isinstance(value, expected_type):
                    raise ValueError(f"El valor para '{key}' debe ser uno de {expected_type}, pero es {type(value)}")
            else:
                if not isinstance(value, expected_type):
                    raise ValueError(f"El valor para '{key}' debe ser {expected_type}, pero es {type(value)}")

        if self._config["small_blind"] >= self._config["big_blind"]:
            raise ValueError("small_blind debe ser menor que big_blind")

        if self._config["action_selection_strategy"] not in ["epsilon_greedy", "softmax"]:
            raise ValueError("action_selection_strategy debe ser 'epsilon_greedy' o 'softmax'")

        if not 0 <= self._config["epsilon"] <= 1:
            raise ValueError("epsilon debe estar entre 0 y 1")

        if self._config["temperature"] <= 0:
            raise ValueError("temperature debe ser mayor que 0")

        if self._config["max_rounds"] <= 0:
            raise ValueError("max_rounds debe ser mayor que 0")

        if self._config["stats_interval"] <= 0:
            raise ValueError("stats_interval debe ser mayor que 0")

        if self._config["game_history_format"] not in ["json", "csv", "pickle"]:
            raise ValueError("game_history_format debe ser 'json', 'csv' o 'pickle'")

        # Validar tamaños de la IA
        for size_key in ["ai_input_size", "ai_hidden_size", "ai_output_size"]:
            if self._config[size_key] <= 0:
                raise ValueError(f"'{size_key}' debe ser mayor que 0")

        # Validar rutas de archivos y carpetas
        for path_key in ["camera_params_path", "card_detection_model_path", "chip_detection_model_path", "ai_model_path"]:
            path = self._config[path_key]
            if not os.path.isfile(path):  # Verifica que sea un archivo
                log.warning(f"La ruta '{path}' para '{path_key}' no es un archivo válido. El programa podría fallar.")

        # Validar existencia de carpetas
        for dir_key in ["log_file"]:  # Agrega aquí las claves de las rutas de carpetas
            dir_path = os.path.dirname(self._config[dir_key])
            if not os.path.isdir(dir_path):
                log.warning(f"La carpeta '{dir_path}' para '{dir_key}' no existe. El programa podría fallar.")

    def apply_environment_variables(self) -> None:
        """
        Sobrescribe la configuración con variables de entorno.
        """
        for key in DEFAULT_CONFIG:
            env_var = os.environ.get(key.upper())
            if env_var:
                # Intenta convertir al tipo correcto
                try:
                    if isinstance(DEFAULT_CONFIG[key], bool):
                        self._config[key] = env_var.lower() == 'true'
                    elif isinstance(DEFAULT_CONFIG[key], int):
                        self._config[key] = int(env_var)
                    elif isinstance(DEFAULT_CONFIG[key], float):
                        self._config[key] = float(env_var)
                    else:
                        self._config[key] = env_var
                    log.info(f"Sobrescribiendo '{key}' con valor de entorno: {env_var}")
                except ValueError:
                    log.warning(f"No se pudo convertir la variable de entorno '{key.upper()}' al tipo correcto. Usando el valor por defecto.")

    @staticmethod
    def from_defaults() -> "Config":
        """
        Crea una instancia de Config con todos los valores por defecto.

        Returns:
            Config: Una instancia de Config con los valores por defecto.
        """
        return Config(DEFAULT_CONFIG)

def load_config(file_path: str) -> Config:
    """
    Carga la configuración desde un archivo JSON.

    Args:
        file_path (str): La ruta al archivo de configuración JSON.

    Returns:
        Config: Una instancia de la clase Config que contiene los parámetros de configuración.

    Raises:
        ConfigLoadError: Si el archivo no se encuentra o si el JSON no es válido.
    """
    try:
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
            config = Config(config_dict)
            return config
    except FileNotFoundError:
        log.error(f"Archivo de configuración no encontrado: {file_path}")
        raise config_exceptions.ConfigLoadError(f"Archivo de configuración no encontrado: {file_path}")
    except json.JSONDecodeError:
        log.error(f"JSON inválido en el archivo de configuración: {file_path}")
        raise config_exceptions.ConfigLoadError(f"JSON inválido en el archivo de configuración: {file_path}")
    except Exception as e:
        log.error(f"Error inesperado al cargar la configuración desde {file_path}: {e}")
        raise config_exceptions.ConfigLoadError(f"Error inesperado: {e}")

def save_config(file_path: str, config: Config) -> None:
    """
    Guarda la configuración en un archivo JSON.

    Args:
        file_path (str): La ruta al archivo de configuración JSON.
        config (Config): Una instancia de la clase Config que contiene los parámetros de configuración a guardar.

    Raises:
        ConfigSaveError: Si hay un IOError al escribir el archivo.
    """
    try:
        # Asegura que el directorio exista
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config.as_dict(), f, indent=4, ensure_ascii=False)
    except IOError as e:
        log.error(f"IOError al escribir la configuración en {file_path}: {e}")
        raise config_exceptions.ConfigSaveError(f"IOError al escribir la configuración en {file_path}: {e}")
    except Exception as e:
        log.error(f"Error inesperado al guardar la configuración en {file_path}: {e}")
        raise config_exceptions.ConfigSaveError(f"Error inesperado: {e}")

# Función para cargar la configuración desde la línea de comandos
def load_config_from_command_line() -> Optional[str]:
    """
    Carga la ruta del archivo de configuración desde la línea de comandos.

    Returns:
        Optional[str]: La ruta al archivo de configuración si se proporciona, None en caso contrario.
    """
    parser = argparse.ArgumentParser(description="Cargar configuración desde un archivo JSON.")
    parser.add_argument("--config_path", type=str, help="Ruta al archivo de configuración JSON.")
    args = parser.parse_args()
    return args.config_path

def handle_config() -> Config:
    """
    Maneja la carga de la configuración, ya sea desde la línea de comandos o desde el archivo por defecto.

    Returns:
        Config: La configuración cargada.
    """
    # Cargar la configuración desde la línea de comandos
    config_path = load_config_from_command_line()

    # Si no se proporciona la ruta, usar la ruta por defecto
    if not config_path:
        config_path = "config/settings.json"

    # Cargar la configuración desde un archivo
    try:
        config = load_config(config_path)
        log.info(f"Configuración cargada desde {config_path}")
        return config
    except config_exceptions.ConfigLoadError as e:
        log.error(f"Error al cargar la configuración: {e}")
        raise  # Relanzar la excepción para que se maneje en el main

# Ejemplo de uso (fuera del módulo principal)
if __name__ == "__main__":
    try:
        # Cargar la configuración
        config = handle_config()

        print("Configuración cargada:", config.as_dict())

        # Acceder a un valor de configuración
        player1_name = config["player1_name"]
        print("Nombre del jugador 1:", player1_name)

        # Modificar un valor de configuración
        config["player2_name"] = "Nuevo Jugador Humano"
        print("Nuevo nombre del jugador 2:", config["player2_name"])

        # Guardar la configuración modificada
        save_config("config/new_settings.json", config)
        print("Configuración guardada en config/new_settings.json")

    except config_exceptions.ConfigLoadError as e:
        print(f"Error al cargar la configuración: {e}")
    except config_exceptions.ConfigSaveError as e:
        print(f"Error al guardar la configuración: {e}")
    except ValueError as e:
        print(f"Error de validación: {e}")