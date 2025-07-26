import torch
import logging
import os

logger = logging.getLogger(__name__)

def save_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filepath: str):
    """
    Guarda el modelo de pesos y el optimizador.
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)
    logger.info(f"Modelo guardado en {filepath}")

def load_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filepath: str):
    """
    Carga el modelo de pesos y el optimizador.
    """
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Modelo cargado desde {filepath}")
    else:
        logger.warning(f"No se encontró el modelo en {filepath}")