import argparse
from typing import Optional
import logging
import os

from config import settings
from exceptions import config_exceptions
from utils.logger import setup_logger

log = setup_logger("config_utils", "logs/config_utils.log")

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

def handle_config() -> settings.Config:
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
        config = settings.load_config(config_path)
        log.info(f"Configuración cargada desde {config_path}")
        return config
    except config_exceptions.ConfigLoadError as e:
        log.error(f"Error al cargar la configuración: {e}")
        raise  # Relanzar la excepción para que se maneje en el main