import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import logging
from typing import List, Tuple, Callable, Optional
from exceptions.ai_exceptions import DatasetLoadError, ModelTrainError, ModelEvaluateError, ModelLoadError
import os
import time

# Configura el logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PokerDataset(Dataset):
    """
    Dataset personalizado para juegos de póker.

    Args:
        data (list): Lista de tuplas (estado, acción, recompensa).
    """
    def __init__(self, data: List[Tuple[List[float], int, float]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state, action, reward = self.data[idx]
        return torch.tensor(state, dtype=torch.float), torch.tensor(action, dtype=torch.long), torch.tensor(reward, dtype=torch.float)

def load_dataset(path: str) -> PokerDataset:
    """
    Carga un dataset desde un archivo.
    """
    try:
        logger.info(f"Cargando dataset desde {path}")
        data = torch.load(path)
        dataset = PokerDataset(data)
        logger.info(f"Dataset cargado correctamente desde {path}")
        return dataset
    except FileNotFoundError as e:
        logger.error(f"Archivo no encontrado al cargar el dataset desde {path}: {e}")
        raise DatasetLoadError(f"No se encontró el archivo {path}") from e
    except Exception as e:
        logger.error(f"Error al cargar el dataset desde {path}: {e}")
        raise DatasetLoadError(f"No se pudo cargar el dataset desde {path}") from e

def train_model(
    model: nn.Module,
    dataset: PokerDataset,
    epochs: int,
    batch_size: int,
    lr: float,
    save_path: str,
    device: str = 'cpu',
    loss_fn: Optional[Callable] = None,
    optimizer_class: Optional[Callable] = None,
    validation_split: float = 0.2,
    patience: int = 10
) -> None:
    """
    Entrena un modelo.
    """
    try:
        # Define la función de pérdida y el optimizador
        loss_fn = loss_fn or nn.MSELoss()
        optimizer_class = optimizer_class or optim.Adam
        optimizer = optimizer_class(model.parameters(), lr=lr)

        # Mueve el modelo al dispositivo
        model.to(device)

        # Divide el dataset en entrenamiento y validación
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Crea los dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Lógica para entrenar el modelo
        logger.info(f"Entrenando modelo en {device} con epochs={epochs}, batch_size={batch_size}, lr={lr}")

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None  # Guarda el estado del mejor modelo
        best_epoch = 0

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for i, (states, actions, rewards) in enumerate(train_dataloader):
                # Mueve los datos al dispositivo
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)

                # Forward pass
                outputs = model(states)
                loss = loss_fn(outputs.gather(1, actions.unsqueeze(1)), rewards.unsqueeze(1))

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Calcula la precisión (opcional)
                _, predicted = torch.max(outputs.data, 1)
                train_total += actions.size(0)
                train_correct += (predicted == actions).sum().item()

            train_loss /= len(train_dataloader)
            train_accuracy = 100 * train_correct / train_total

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for states, actions, rewards in val_dataloader:
                    # Mueve los datos al dispositivo
                    states = states.to(device)
                    actions = actions.to(device)
                    rewards = rewards.to(device)

                    outputs = model(states)
                    loss = loss_fn(outputs.gather(1, actions.unsqueeze(1)), rewards.unsqueeze(1))
                    val_loss += loss.item()

                    # Calcula la precisión (opcional)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += actions.size(0)
                    val_correct += (predicted == actions).sum().item()

            val_loss /= len(val_dataloader)
            val_accuracy = 100 * val_correct / val_total

            logger.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()  # Guarda el estado del modelo
                best_epoch = epoch + 1

            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    logger.info(f"Early stopping triggered after {patience} epochs without improvement. Best epoch: {best_epoch}, Best validation loss: {best_val_loss:.4f}")
                    break

        # Guarda el mejor modelo con nombre dinámico
        if best_model_state is not None:
            # Agrega un timestamp al nombre del archivo para evitar sobreescrituras
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename, ext = os.path.splitext(save_path)
            best_model_path = f"{filename}_{timestamp}_best{ext}"
            torch.save(best_model_state, best_model_path)
            logger.info(f"Mejor modelo guardado en {best_model_path}")
        else:
            logger.warning("No se guardó ningún modelo porque no hubo mejora en la validación.")

        logger.info(f"Entrenamiento completado. Mejor validation loss: {best_val_loss:.4f}")

    except Exception as e:
        logger.error(f"Error al entrenar el modelo: {e}")
        raise ModelTrainError(f"No se pudo entrenar el modelo") from e

def load_model(model: nn.Module, filepath: str, device: str = 'cpu') -> nn.Module:
    """
    Carga un modelo desde un archivo y lo configura en modo de evaluación.
    """
    try:
        model.load_state_dict(torch.load(filepath, map_location=device))
        model.to(device)
        model.eval()
        logger.info(f"Modelo cargado desde {filepath} y puesto en modo de evaluación en {device}")
        return model
    except FileNotFoundError as e:
        logger.error(f"Archivo no encontrado al cargar el modelo desde {filepath}: {e}")
        raise ModelLoadError(f"No se encontró el archivo {filepath}") from e
    except Exception as e:
        logger.error(f"Error al cargar el modelo desde {filepath}: {e}")
        raise ModelLoadError(f"No se pudo cargar el modelo desde {filepath}") from e

def evaluate_model(model: nn.Module, validation_data: PokerDataset, device: str = 'cpu') -> float:
    """
    Evalúa un modelo.
    """
    try:
        model.to(device)
        model.eval()
        dataloader = DataLoader(validation_data, batch_size=32)
        correct = 0
        total = 0
        with torch.no_grad():
            for states, actions, rewards in dataloader:
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)

                outputs = model(states)
                _, predicted = torch.max(outputs.data, 1)
                total += actions.size(0)
                correct += (predicted == actions).sum().item()

        accuracy = 100 * correct / total
        logger.info(f"Precisión del modelo en los datos de validación: {accuracy:.2f}%")
        return accuracy

    except Exception as e:
        logger.error(f"Error al evaluar el modelo: {e}")
        raise ModelEvaluateError(f"No se pudo evaluar el modelo") from e