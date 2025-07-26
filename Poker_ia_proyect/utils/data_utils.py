import json
import csv
import pickle
import logging

def load_data(file_path):
    """
    Carga datos desde un archivo JSON.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: El archivo {file_path} no fue encontrado.")
        return None
    except json.JSONDecodeError:
        print(f"Error: El archivo {file_path} no contiene JSON válido.")
        return None

def save_game_history(game_history: list, filename: str, format: str = "json") -> None:
    """Exporta game_history al final en formato .json, .csv o .pkl."""
    try:
        if format == "json":
            with open(filename + ".json", "w") as f:
                json.dump(game_history, f, indent=4)
            logging.info(f"Historial del juego guardado en {filename}.json")
        elif format == "csv":
            with open(filename + ".csv", "w", newline="") as f:
                writer = csv.writer(f)
                if game_history:
                    header = game_history[0].keys()
                    writer.writerow(header)
                    for data in game_history:
                        writer.writerow(data.values())
            logging.info(f"Historial del juego guardado en {filename}.csv")
        elif format == "pkl":
            with open(filename + ".pkl", "wb") as f:
                pickle.dump(game_history, f)
            logging.info(f"Historial del juego guardado en {filename}.pkl")
        else:
            raise ValueError("Formato de archivo no soportado. Use json, csv o pkl.")
    except Exception as e:
        logging.error(f"Error al guardar el historial del juego: {e}")