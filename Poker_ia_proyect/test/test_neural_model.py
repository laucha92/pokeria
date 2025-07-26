import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from ai.neural_model import PokerNet, ModelSaveError, ModelLoadError

# Define una fixture para crear una instancia del modelo
@pytest.fixture
def model():
    return PokerNet(input_size=10, hidden_size=20, output_size=3)

# Define una fixture para crear un archivo temporal para guardar el modelo
@pytest.fixture
def model_file(tmpdir):
    return os.path.join(tmpdir, "test_model.pth")

# Prueba que el modelo se guarda correctamente
def test_save_model(model, model_file):
    model.save(model_file)
    assert os.path.exists(model_file)

# Prueba que el modelo se carga correctamente
def test_load_model(model, model_file):
    model.save(model_file)
    loaded_model = PokerNet(input_size=10, hidden_size=20, output_size=3)
    loaded_model.load(model_file)
    assert isinstance(loaded_model, PokerNet)
    assert loaded_model.training == False  # Verifica que está en modo de evaluación

# Prueba que se lanza una excepción si el archivo no existe
def test_load_model_file_not_found():
    with pytest.raises(ModelLoadError):
        loaded_model = PokerNet(input_size=10, hidden_size=20, output_size=3)
        loaded_model.load("non_existent_file.pth")

# Prueba que la propagación hacia adelante funciona correctamente
def test_forward_pass(model):
    input_tensor = torch.randn(1, 10)
    output_tensor = model(input_tensor)
    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.shape == (1, 3)