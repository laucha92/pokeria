import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from exceptions.ai_exceptions import ModelSaveError, ModelLoadError
import os

# Configura el logger (se puede configurar desde fuera)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

class PokerNet(nn.Module):
    """
    Red neuronal compacta y eficiente para aproximar la función de estrategia en juegos de póker,
    con soporte opcional para dispositivos GPU y configuración flexible del logging.

    Args:
        input_size (int): Tamaño de la capa de entrada (número de características del estado del juego).
        hidden_size (int): Tamaño de la capa oculta.
        output_size (int): Tamaño de la capa de salida (número de acciones posibles).
        device (str, optional): Dispositivo para ejecutar el modelo ('cpu' o 'cuda'). Por defecto es 'cpu'.

    Ejemplo:
        model = PokerNet(input_size=100, hidden_size=200, output_size=4, device='cuda')
        input_tensor = torch.randn(1, 100).to(model.device)  # Ejemplo de entrada en el dispositivo correcto
        output_tensor = model(input_tensor)
        print(output_tensor)
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, device: str = 'cpu'):
        """
        Inicializa la red neuronal.

        Args:
            input_size (int): Tamaño de la capa de entrada.
            hidden_size (int): Tamaño de la capa oculta.
            output_size (int): Tamaño de la capa de salida.
            device (str, optional): Dispositivo para ejecutar el modelo ('cpu' o 'cuda'). Por defecto es 'cpu'.
        """
        super(PokerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.device = device
        self.to(device)  # Mueve el modelo al dispositivo especificado

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propaga los datos a través de la red.

        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            torch.Tensor: Tensor de salida.
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self, filepath: str):
        """
        Guarda el modelo en un archivo, creando el directorio si no existe.

        Args:
            filepath (str): Ruta del archivo donde se guardará el modelo.

        Raises:
            ModelSaveError: Si ocurre un error al guardar el modelo.
        """
        try:
            # Crea el directorio si no existe
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            # Guarda el modelo en la CPU antes de guardar
            self.to('cpu')
            torch.save(self.state_dict(), filepath)
            # Mueve el modelo de vuelta al dispositivo original
            self.to(self.device)
            logger.info(f"Modelo guardado en {filepath}")
        except Exception as e:
            logger.error(f"Error al guardar el modelo en {filepath}: {e}")
            raise ModelSaveError(f"No se pudo guardar el modelo en {filepath}") from e

    def load(self, filepath: str):
        """
        Carga el modelo desde un archivo y lo pone en modo de evaluación.

        Args:
            filepath (str): Ruta del archivo desde donde se cargará el modelo.

        Raises:
            ModelLoadError: Si ocurre un error al cargar el modelo.
        """
        try:
            # Carga el modelo mapeándolo al dispositivo actual
            self.load_state_dict(torch.load(filepath, map_location=self.device))
            self.eval()  # Pone el modelo en modo de evaluación
            logger.info(f"Modelo cargado desde {filepath} y puesto en modo de evaluación en el dispositivo {self.device}.")
        except FileNotFoundError as e:
            logger.error(f"Archivo no encontrado al cargar el modelo desde {filepath}: {e}")
            raise ModelLoadError(f"No se encontró el archivo {filepath}") from e
        except RuntimeError as e:
            logger.error(f"Error de runtime al cargar el modelo desde {filepath}: {e}")
            raise ModelLoadError(f"Incompatibilidad al cargar el modelo desde {filepath}") from e
        except Exception as e:
            logger.error(f"Error desconocido al cargar el modelo desde {filepath}: {e}")
            raise ModelLoadError(f"Error desconocido al cargar el modelo desde {filepath}") from e

# Ejemplo de uso (puedes agregar esto en un archivo test_neural_model.py)
if __name__ == '__main__':
    try:
        # 1. Crear una instancia del modelo
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = PokerNet(input_size=10, hidden_size=20, output_size=3, device=device)

        # 2. Guardar el modelo
        model.save("models/test_model.pth")

        # 3. Cargar el modelo
        loaded_model = PokerNet(input_size=10, hidden_size=20, output_size=3, device=device)  # Debe tener la misma arquitectura
        loaded_model.load("models/test_model.pth")

        # 4. Verificar que el modelo está en modo de evaluación
        print("El modelo está en modo de evaluación:", loaded_model.training == False)
        print("El modelo está en el dispositivo:", loaded_model.device)

        print("Modelo cargado y verificado con éxito.")

    except ModelSaveError as e:
        print(f"Error al guardar el modelo: {e}")
    except ModelLoadError as e:
        print(f"Error al cargar el modelo: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")