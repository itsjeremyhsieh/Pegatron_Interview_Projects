import unittest
from unittest.mock import patch, MagicMock
import sys
import builtins

import run_benchmark  # Replace with the actual filename of your script

class TestBenchmarkPipeline(unittest.TestCase):

    @patch("run_benchmark.load_mmmu_dataset")
    @patch("run_benchmark.BenchmarkModel")
    @patch("run_benchmark.evaluate_model")
    def test_main_pipeline(self, mock_evaluate, mock_model_class, mock_load_dataset):
        # Mock dataset
        mock_dataset = [{"question": "Q1", "options": ["A","B","C","D"], "answer": "A"}]
        mock_load_dataset.return_value = mock_dataset

        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        mock_evaluate.return_value = {"accuracy": 0.9, "correct_str": "A,A,A"}
        
        test_args = ["benchmark_pipeline.py", "--model", "gpt-4o-mini", "--max_samples", "5", "--subject", "Accounting"]
        with patch.object(sys, "argv", test_args):
            with patch.object(builtins, "print") as mock_print:
                run_benchmark.main()
                mock_load_dataset.assert_called_once_with(subject="Accounting")
                mock_model_class.assert_called_once_with("gpt-4o-mini")
                mock_evaluate.assert_called_once_with(mock_model, mock_dataset, max_samples=5)
                printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list)
                self.assertIn("accuracy: 0.9", printed)
                self.assertIn("A,A,A", printed)

if __name__ == "__main__":
    unittest.main()
