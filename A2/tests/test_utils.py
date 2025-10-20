import unittest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import compute_accuracy


class TestComputeAccuracy(unittest.TestCase):
    """Test suite for compute_accuracy function"""
    
    def test_perfect_accuracy(self):
        """Test accuracy when all predictions are correct"""
        y_true = ['A', 'B', 'C', 'D']
        y_pred = ['A', 'B', 'C', 'D']
        
        result = compute_accuracy(y_true, y_pred)
        
        self.assertEqual(result, 1.0, "Perfect predictions should give 1.0 accuracy")
    
    def test_zero_accuracy(self):
        """Test accuracy when all predictions are incorrect"""
        y_true = ['A', 'B', 'C', 'D']
        y_pred = ['B', 'C', 'D', 'A']
        
        result = compute_accuracy(y_true, y_pred)
        
        self.assertEqual(result, 0.0, "All wrong predictions should give 0.0 accuracy")
    
    def test_fifty_percent_accuracy(self):
        """Test accuracy with 50% correct predictions"""
        y_true = ['A', 'B', 'C', 'D']
        y_pred = ['A', 'B', 'X', 'Y']
        
        result = compute_accuracy(y_true, y_pred)
        
        self.assertEqual(result, 0.5, "50% correct should give 0.5 accuracy")
    
    def test_partial_accuracy(self):
        """Test accuracy with partial correct predictions"""
        y_true = ['A', 'B', 'C', 'D', 'E']
        y_pred = ['A', 'X', 'C', 'Y', 'E']
        
        result = compute_accuracy(y_true, y_pred)
        
        self.assertEqual(result, 0.6, "3 out of 5 correct should give 0.6 accuracy")
    
    def test_single_element_correct(self):
        """Test accuracy with single element that is correct"""
        y_true = ['A']
        y_pred = ['A']
        
        result = compute_accuracy(y_true, y_pred)
        
        self.assertEqual(result, 1.0, "Single correct prediction should give 1.0")
    
    def test_single_element_incorrect(self):
        """Test accuracy with single element that is incorrect"""
        y_true = ['A']
        y_pred = ['B']
        
        result = compute_accuracy(y_true, y_pred)
        
        self.assertEqual(result, 0.0, "Single incorrect prediction should give 0.0")
    
    def test_numeric_labels(self):
        """Test accuracy with numeric labels"""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3, 4, 5]
        
        result = compute_accuracy(y_true, y_pred)
        
        self.assertEqual(result, 1.0, "Should work with numeric labels")
    
    def test_mixed_numeric_labels(self):
        """Test accuracy with mixed numeric predictions"""
        y_true = [1, 2, 3, 4]
        y_pred = [1, 0, 3, 0]
        
        result = compute_accuracy(y_true, y_pred)
        
        self.assertEqual(result, 0.5, "Should correctly compute accuracy for numeric labels")
    