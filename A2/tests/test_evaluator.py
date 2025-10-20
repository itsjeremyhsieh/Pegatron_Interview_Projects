from evaluator import normalize_to_letter, evaluate_model
import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestNormalizeToLetter(unittest.TestCase):
    """Test suite for normalize_to_letter function"""

    def test_single_letter_uppercase(self):
        """Test extraction of single uppercase letter"""
        self.assertEqual(normalize_to_letter("A"), "A")
        self.assertEqual(normalize_to_letter("B"), "B")
        self.assertEqual(normalize_to_letter("J"), "J")

    def test_single_letter_lowercase(self):
        """Test extraction of single lowercase letter"""
        self.assertEqual(normalize_to_letter("a"), "A")
        self.assertEqual(normalize_to_letter("b"), "B")
        self.assertEqual(normalize_to_letter("j"), "J")

    def test_letter_with_text(self):
        """Test extraction of letter from text"""
        self.assertEqual(normalize_to_letter("The answer is A"), "A")
        self.assertEqual(normalize_to_letter("Choose B"), "B")
        self.assertEqual(normalize_to_letter("Option C is correct"), "C")

    def test_empty_string(self):
        """Test handling of empty string"""
        self.assertEqual(normalize_to_letter(""), "")

    def test_none_input(self):
        """Test handling of None input"""
        self.assertEqual(normalize_to_letter(None), "")

    def test_whitespace_handling(self):
        """Test handling of whitespace"""
        self.assertEqual(normalize_to_letter("  A  "), "A")
        self.assertEqual(normalize_to_letter("\nB\n"), "B")


class TestEvaluateModel(unittest.TestCase):
    """Test suite for evaluate_model function"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock()
        self.sample_dataset = [
            {
                "question": "What is 2+2?",
                "image": None,
                "options": ["3", "4", "5"],
                "label": "B"
            },
            {
                "question": "What is 3+3?",
                "image": None,
                "options": ["5", "6", "7"],
                "label": "B"
            }
        ]

    @patch('evaluator.tqdm')
    @patch('builtins.print')
    def test_basic_evaluation(self, mock_print, mock_tqdm):
        """Test basic evaluation with correct predictions"""
        mock_tqdm.return_value = MagicMock()
        self.mock_model.predict.side_effect = ["B", "B"]

        result = evaluate_model(
            self.mock_model, self.sample_dataset, max_samples=2)

        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["correct_str"], "2/2")

    @patch('evaluator.tqdm')
    @patch('builtins.print')
    def test_partial_correct(self, mock_print, mock_tqdm):
        """Test evaluation with some incorrect predictions"""
        mock_tqdm.return_value = MagicMock()
        self.mock_model.predict.side_effect = ["B", "A"]

        result = evaluate_model(
            self.mock_model, self.sample_dataset, max_samples=2)

        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["correct_str"], "1/2")

    @patch('evaluator.tqdm')
    @patch('builtins.print')
    def test_all_incorrect(self, mock_print, mock_tqdm):
        """Test evaluation with all incorrect predictions"""
        mock_tqdm.return_value = MagicMock()
        self.mock_model.predict.side_effect = ["A", "A"]

        result = evaluate_model(
            self.mock_model, self.sample_dataset, max_samples=2)

        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["correct_str"], "0/2")

    @patch('evaluator.tqdm')
    @patch('builtins.print')
    def test_max_samples_limit(self, mock_print, mock_tqdm):
        """Test that max_samples limits the number of processed samples"""
        mock_tqdm.return_value = MagicMock()
        self.mock_model.predict.return_value = "B"

        result = evaluate_model(
            self.mock_model, self.sample_dataset, max_samples=1)

        self.assertEqual(len(result["correct_str"].split("/")[1]), 1)
        self.assertEqual(self.mock_model.predict.call_count, 1)

    @patch('evaluator.tqdm')
    @patch('builtins.print')
    def test_skips_samples_without_options(self, mock_print, mock_tqdm):
        """Test that samples without options are skipped"""
        mock_tqdm.return_value = MagicMock()
        dataset = [
            {
                "question": "Question?",
                "image": None,
                "options": [],
                "label": "A"
            },
            {
                "question": "Another?",
                "image": None,
                "options": ["A", "B"],
                "label": "A"
            }
        ]
        self.mock_model.predict.return_value = "A"

        result = evaluate_model(self.mock_model, dataset, max_samples=2)

        # Should only process the second sample
        self.assertEqual(self.mock_model.predict.call_count, 1)

    @patch('evaluator.tqdm')
    @patch('builtins.print')
    def test_integer_label(self, mock_print, mock_tqdm):
        """Test handling of integer labels"""
        mock_tqdm.return_value = MagicMock()
        dataset = [
            {
                "question": "Question?",
                "image": None,
                "options": ["Option A", "Option B", "Option C"],
                "label": 1  # Index 1 = "B"
            }
        ]
        self.mock_model.predict.return_value = "B"

        result = evaluate_model(self.mock_model, dataset, max_samples=1)

        self.assertEqual(result["accuracy"], 1.0)

    @patch('evaluator.tqdm')
    @patch('builtins.print')
    def test_model_predict_called_correctly(self, mock_print, mock_tqdm):
        """Test that model.predict is called with correct arguments"""
        mock_tqdm.return_value = MagicMock()
        self.mock_model.predict.return_value = "B"

        evaluate_model(self.mock_model, self.sample_dataset, max_samples=1)

        # Check first call
        call_args = self.mock_model.predict.call_args[0]
        self.assertEqual(call_args[0], "What is 2+2?")
        self.assertIsNone(call_args[1])
        self.assertEqual(call_args[2], ["3", "4", "5"])

    @patch('evaluator.tqdm')
    @patch('builtins.print')
    def test_return_structure(self, mock_print, mock_tqdm):
        """Test that return value has correct structure"""
        mock_tqdm.return_value = MagicMock()
        self.mock_model.predict.return_value = "B"

        result = evaluate_model(
            self.mock_model, self.sample_dataset, max_samples=1)

        self.assertIn("accuracy", result)
        self.assertIn("correct_str", result)
        self.assertIsInstance(result["accuracy"], float)
        self.assertIsInstance(result["correct_str"], str)


if __name__ == '__main__':
    unittest.main(verbosity=2)
