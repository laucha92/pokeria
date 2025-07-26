import unittest
import numpy as np
from typing import List, Dict, Any
from unittest.mock import patch
import io
import contextlib

from features import extract_features, _normalize_with_nan_check, DEFAULT_NON_NUMERIC_VALUE, _calculate_has_pair, _calculate_has_strong_draw, _calculate_board_texture, _calculate_has_top_pair, _calculate_has_overcards, _evaluate_hand_strength, export_features

class TestFeatureExtraction(unittest.TestCase):
    def test_normalize_value(self):
        self.assertEqual(_normalize_with_nan_check(50, 0, 100), 0.5)
        self.assertEqual(_normalize_with_nan_check(0, 0, 100), 0.0)
        self.assertEqual(_normalize_with_nan_check(100, 0, 100), 1.0)
        self.assertEqual(_normalize_with_nan_check(50, 50, 50), 0.0)  # Test división por cero
        self.assertEqual(_normalize_with_nan_check(np.nan, 0, 100), DEFAULT_NON_NUMERIC_VALUE)  # Test NaN
        self.assertEqual(_normalize_with_nan_check(np.inf, 0, 100), DEFAULT_NON_NUMERIC_VALUE)  # Test Inf

    def test_extract_features_minimal(self):
        game_state = {'pot': 50, 'community_cards': [], 'round_num': 3, 'current_bet': 10, 'min_raise': 5, 'opponents': []}
        player_state = {'chips': 100, 'hand': ['A', 'K'], 'raises': 1, 'bets': 2, 'calls': 3}
        features = extract_features(game_state, player_state)
        self.assertEqual(len(features), 24)  # Verifica que se extraen todas las features
        self.assertIsInstance(features, list)  # Verifica que el resultado es una lista

    def test_extract_features_custom_config(self):
        game_state = {'pot': 50, 'community_cards': [], 'round_num': 3, 'current_bet': 10, 'min_raise': 5, 'opponents': []}
        player_state = {'chips': 100, 'hand': ['A', 'K'], 'raises': 1, 'bets': 2, 'calls': 3}
        feature_config = {
            "normalize_chips": False,
            "include_sections": ["player", "game"],
            "custom_features": {
                "test_feature": {
                    "func": lambda gs, ps: 1.0
                }
            }
        }
        features = extract_features(game_state, player_state, feature_config=feature_config, verbose=True)
        self.assertIsInstance(features, list)  # Verifica que el resultado es una lista
        self.assertEqual(len(features), 21)  # Verifica que se extraen todas las features

    def test_extract_features_missing_fields(self):
        game_state = {'pot': 50, 'community_cards': [], 'round_num': 3}  # Faltan campos
        player_state = {'chips': 100, 'hand': ['A', 'K']}
        features = extract_features(game_state, player_state)
        self.assertEqual(len(features), 24)  # Debería seguir extrayendo, aunque con warnings

    def test_extract_features_custom_exception(self):
        game_state = {'pot': 50, 'community_cards': [], 'round_num': 3, 'current_bet': 10, 'min_raise': 5, 'opponents': []}
        player_state = {'chips': 100, 'hand': ['A', 'K'], 'raises': 1, 'bets': 2, 'calls': 3}
        feature_config = {
            "custom_features": {
                "test_feature": {
                    "func": lambda gs, ps: 1.0 / 0.0  # Lanza excepción
                }
            }
        }
        features = extract_features(game_state, player_state, feature_config=feature_config)
        # Debería seguir extrayendo, aunque con error en la custom feature
        self.assertEqual(len(features), 24)

    def test_normalized_features_range(self):
        game_state = {'pot': 5000, 'community_cards': [], 'round_num': 3, 'current_bet': 1000, 'min_raise': 500, 'opponents': []}
        player_state = {'chips': 10000, 'hand': ['A', 'K'], 'raises': 1, 'bets': 2, 'calls': 3}
        features = extract_features(game_state, player_state)
        for feature in features:
            self.assertTrue(0.0 <= feature <= 1.0 or not isinstance(feature, float))  # Verifica el rango [0, 1]

    def test_extract_features_edge_cases(self):
        game_state = {'pot': 0, 'community_cards': [], 'round_num': 0, 'current_bet': 0, 'min_raise': 0, 'opponents': []}
        player_state = {'chips': 0, 'hand': [], 'raises': 0, 'bets': 0, 'calls': 0, 'is_all_in': True}
        features = extract_features(game_state, player_state)
        self.assertEqual(len(features), 24)

    def test_extract_features_invalid_custom(self):
        game_state = {'pot': 50, 'community_cards': [], 'round_num': 3, 'current_bet': 10, 'min_raise': 5, 'opponents': []}
        player_state = {'chips': 100, 'hand': ['A', 'K'], 'raises': 1, 'bets': 2, 'calls': 3}
        feature_config = {
            "custom_features": {
                "invalid_feature": "not a function"
            }
        }
        features = extract_features(game_state, player_state, feature_config=feature_config)
        self.assertEqual(len(features), 24)

    def test_extract_features_partial_extraction(self):
        game_state = {'pot': 50, 'community_cards': [], 'round_num': 3, 'current_bet': 10, 'min_raise': 5, 'opponents': []}
        player_state = {'chips': 100, 'hand': ['A', 'K'], 'raises': 1, 'bets': 2, 'calls': 3}
        feature_config = {
            "include_sections": ["player"]
        }
        features = extract_features(game_state, player_state, feature_config=feature_config)
        self.assertEqual(len(features), 5)

    def test_extract_features_return_named(self):
        game_state = {'pot': 50, 'community_cards': [], 'round_num': 3, 'current_bet': 10, 'min_raise': 5, 'opponents': []}
        player_state = {'chips': 100, 'hand': ['A', 'K'], 'raises': 1, 'bets': 2, 'calls': 3}
        named_features = extract_features(game_state, player_state, return_named=True)
        self.assertIsInstance(named_features, dict)
        self.assertEqual(len(named_features), 24)
        self.assertIn("chips", named_features)

    def test_extract_features_edge_cases_bet_ratios(self):
        game_state = {'pot': 50, 'community_cards': [], 'round_num': 3, 'current_bet': 0, 'min_raise': 5, 'opponents': []}
        player_state = {'chips': 100, 'hand': ['A', 'K'], 'raises': 1, 'bets': 2, 'calls': 3, 'is_all_in': True}
        features = extract_features(game_state, player_state)
        self.assertEqual(len(features), 24)

    def test_extract_features_types(self):
        game_state = {'pot': 50, 'community_cards': [], 'round_num': 3, 'current_bet': 10, 'min_raise': 5, 'opponents': []}
        player_state = {'chips': 100, 'hand': ['A', 'K'], 'raises': 1, 'bets': 2, 'calls': 3}
        features = extract_features(game_state, player_state)
        for feature in features:
            self.assertIsInstance(feature, (int, float))

    def test_export_features_table(self):
        feature_names = ["feature1", "feature2"]
        features = [0.5, 0.75]
        include_sections = ["player", "game"]
        
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            export_features(feature_names, features, include_sections, output_format="table")
            captured_output = mock_stdout.getvalue().strip()

            self.assertTrue("Feature" in captured_output)
            self.assertTrue("Value" in captured_output)
            self.assertTrue("feature1" in captured_output)
            self.assertTrue("feature2" in captured_output)
            self.assertTrue("0.5" in captured_output)
            self.assertTrue("0.75" in captured_output)

    def test_export_features_csv(self):
        feature_names = ["feature1", "feature2"]
        features = [0.5, 0.75]
        include_sections = ["player", "game"]
        
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            export_features(feature_names, features, include_sections, output_format="csv")
            captured_output = mock_stdout.getvalue().strip()

            self.assertEqual(captured_output, "feature1,feature2\n0.5,0.75")

    def test_export_features_json(self):
        feature_names = ["feature1", "feature2"]
        features = [0.5, 0.75]
        include_sections = ["player", "game"]
        
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            export_features(feature_names, features, include_sections, output_format="json")
            captured_output = mock_stdout.getvalue().strip()

            self.assertTrue('"feature": "feature1"' in captured_output)
            self.assertTrue('"value": 0.5' in captured_output)
            self.assertTrue('"feature": "feature2"' in captured_output)
            self.assertTrue('"value": 0.75' in captured_output)

    def test_export_features_invalid_format(self):
        feature_names = ["feature1", "feature2"]
        features = [0.5, 0.75]
        include_sections = ["player", "game"]
        
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            export_features(feature_names, features, include_sections, output_format="invalid")
            captured_output = mock_stdout.getvalue().strip()

            self.assertEqual(captured_output, "")

if __name__ == '__main__':
    unittest.main()