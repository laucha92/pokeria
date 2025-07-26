import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as t

class SimpleWeightModel(nn.Module):
    """
    Modelo de red neuronal para ponderar diferentes estrategias en Poker Texas Hold'em Heads Up.

    Características:
    - Entrada: Un vector de tamaño configurable `state_size`, que representa el estado actual del juego.
    - Salida: Un vector de tamaño `num_strategies`, que representa los pesos asignados a cada estrategia.
    - Arquitectura: Dos capas ocultas con BatchNorm y Dropout, conexiones skip opcionales e ingeniería de características.
    - Incluye normalización de entrada, regularización L2 y soporte para entrada 1D o batch.
    - Compatible con GPU.
    """

    def __init__(self, num_strategies: int, state_size: int, dropout_rate: float = 0.1, l2_reg: float = 0.001,
                 num_skip_features: int = 0,
                 feature_combinations: t.Optional[t.List[t.Tuple[int, int]]] = None):
        """
        Inicializa el modelo.

        Args:
            num_strategies: Número de estrategias a ponderar (salida).
            state_size: Tamaño del vector de estado de entrada.
            dropout_rate: Probabilidad de dropout para regularización.
            l2_reg: Coeficiente de regularización L2.
            num_skip_features: Número de características de entrada para conexión skip.
            feature_combinations: Lista de tuplas (i, j) para multiplicar características.
        """
        super().__init__()
        self.state_size = state_size
        self.num_strategies = num_strategies
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.num_skip_features = num_skip_features
        self.feature_combinations = feature_combinations or []

        # Normalización de entrada
        self.input_norm = nn.LayerNorm(state_size)

        # Capas principales
        self.fc1 = nn.Linear(state_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_strategies)

        # Inicialización de pesos
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 1.0 / num_strategies)

        # Conexión skip opcional
        if num_skip_features > 0:
            self.skip_fc = nn.Linear(num_skip_features, num_strategies)
            nn.init.xavier_uniform_(self.skip_fc.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pasa un tensor de entrada a través de la red neuronal.

        Args:
            x: Tensor de entrada [batch_size, state_size] o [state_size].

        Returns:
            Tensor de salida [batch_size, num_strategies].
        """
        # Soporte para entrada 1D
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Normalización de entrada
        x = self.input_norm(x)

        # Ingeniería de características combinadas
        for i, j in self.feature_combinations:
            x[:, i] = x[:, i] * x[:, j]

        # Forward principal
        x_main = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x_main = self.dropout2(F.relu(self.bn2(self.fc2(x_main))))
        x_out = self.fc3(x_main)

        # Conexión skip opcional
        if self.num_skip_features > 0:
            skip = self.skip_fc(x[:, :self.num_skip_features])
            x_out = x_out + skip

        return x_out

    def l2_loss(self) -> torch.Tensor:
        """
        Calcula la pérdida L2 para regularización.
        """
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param)
        return self.l2_reg * l2_loss

    def save(self, path: str):
        """
        Guarda los pesos del modelo.
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location=None):
        """
        Carga los pesos del modelo.
        """
        self.load_state_dict(torch.load(path, map_location=map_location))

# --- Test básico y ejemplo de uso ---
if __name__ == '__main__':
    num_strategies = 3
    state_size = 10
    dropout_rate = 0.1
    l2_reg = 0.001
    num_skip_features = 3
    feature_combinations = [(0, 1), (2, 3)]

    model = SimpleWeightModel(num_strategies, state_size, dropout_rate, l2_reg,
                              num_skip_features=num_skip_features,
                              feature_combinations=feature_combinations)

    print("Arquitectura del modelo:")
    print(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"\nUsando dispositivo: {device}")

    batch_size = 2
    input_tensor = torch.randn(batch_size, state_size).to(device)
    output_tensor = model(input_tensor)

    print("\nTamaño del tensor de salida:")
    print(output_tensor.size())
    print("\nEjemplo de salida del modelo:")
    print(output_tensor)

    # Test unitario automático
    assert output_tensor.shape == (batch_size, num_strategies), "La salida no tiene la forma esperada."

    # Guardar y cargar modelo (test)
    model.save("test_weights.pth")
    model.load("test_weights.pth", map_location=device)
    print("\nGuardado y carga de pesos exitosos.")

    print("\nPrueba completada.")