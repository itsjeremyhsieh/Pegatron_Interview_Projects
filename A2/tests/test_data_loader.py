import unittest
from unittest.mock import Mock, patch, MagicMock, call
from datasets import Dataset

class TestLoadMmmuDataset(unittest.TestCase):
    """Test suite for load_mmmu_dataset function"""
    
    @patch('data_loader.load_dataset')
    def test_load_dataset_called_with_correct_params(self, mock_load_dataset):
        """Test that load_dataset is called with correct parameters"""
        from data_loader import load_mmmu_dataset
        
        mock_dataset = Mock()
        mock_dataset.cast_column.return_value = mock_dataset
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset
        
        load_mmmu_dataset(split="validation", subject="Accounting")
        
        mock_load_dataset.assert_called_once_with("MMMU/MMMU", "Accounting", split="validation")
    
    @patch('data_loader.load_dataset')
    def test_default_parameters(self, mock_load_dataset):
        """Test function uses default parameters correctly"""
        from data_loader import load_mmmu_dataset
        
        mock_dataset = Mock()
        mock_dataset.cast_column.return_value = mock_dataset
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset
        
        load_mmmu_dataset()
        
        mock_load_dataset.assert_called_once_with("MMMU/MMMU", "Accounting", split="validation")
    
    @patch('data_loader.load_dataset')
    def test_cast_column_success(self, mock_load_dataset):
        """Test successful casting of image_1 column"""
        from data_loader import load_mmmu_dataset
        
        mock_dataset = Mock()
        mock_dataset.cast_column.return_value = mock_dataset
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset
        
        load_mmmu_dataset()
        
        mock_dataset.cast_column.assert_called_once()
    
    @patch('builtins.print')
    @patch('data_loader.load_dataset')
    def test_cast_column_exception_handling(self, mock_load_dataset, mock_print):
        """Test that exceptions during column casting are handled gracefully"""
        from data_loader import load_mmmu_dataset
        
        mock_dataset = Mock()
        mock_dataset.cast_column.side_effect = Exception("Cast failed")
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset
        
        result = load_mmmu_dataset()
        
        # Should print warning but continue
        warning_printed = any("Warning: could not cast 'image_1' column" in str(call_args) 
                              for call_args in mock_print.call_args_list)
        self.assertTrue(warning_printed)
        self.assertIsNotNone(result)
    
    @patch('data_loader.load_dataset')
    def test_preprocess_valid_options(self, mock_load_dataset):
        """Test preprocessing with valid options string"""
        from data_loader import load_mmmu_dataset
        
        # Create a mock dataset with sample data
        sample_data = {
            "id": ["q1"],
            "question": ["What is 2+2?"],
            "image_1": [None],
            "options": ["['A', 'B', 'C', 'D']"],
            "answer": ["A"]
        }
        
        mock_dataset = Dataset.from_dict(sample_data)
        mock_load_dataset.return_value = mock_dataset
        
        with patch.object(mock_dataset, 'cast_column', return_value=mock_dataset):
            result = load_mmmu_dataset()
        
        self.assertEqual(result[0]["options"], ['A', 'B', 'C', 'D'])
        self.assertEqual(result[0]["label"], "A")
    
    @patch('data_loader.load_dataset')
    def test_preprocess_invalid_options_syntax_error(self, mock_load_dataset):
        """Test preprocessing with invalid options string (SyntaxError)"""
        from data_loader import load_mmmu_dataset
        
        sample_data = {
            "id": ["q1"],
            "question": ["What is 2+2?"],
            "image_1": [None],
            "options": ["[invalid syntax"],
            "answer": ["A"]
        }
        
        mock_dataset = Dataset.from_dict(sample_data)
        mock_load_dataset.return_value = mock_dataset
        
        with patch.object(mock_dataset, 'cast_column', return_value=mock_dataset):
            result = load_mmmu_dataset()
        
        self.assertEqual(result[0]["options"], [])
    
    @patch('data_loader.load_dataset')
    def test_preprocess_non_list_options(self, mock_load_dataset):
        """Test preprocessing when options parse to non-list type"""
        from data_loader import load_mmmu_dataset
        
        sample_data = {
            "id": ["q1"],
            "question": ["What is 2+2?"],
            "image_1": [None],
            "options": ["'not a list'"],
            "answer": ["A"]
        }
        
        mock_dataset = Dataset.from_dict(sample_data)
        mock_load_dataset.return_value = mock_dataset
        
        with patch.object(mock_dataset, 'cast_column', return_value=mock_dataset):
            result = load_mmmu_dataset()
        
        self.assertEqual(result[0]["options"], [])
    
    @patch('data_loader.load_dataset')
    def test_preprocess_missing_options_field(self, mock_load_dataset):
        """Test preprocessing when options field is missing"""
        from data_loader import load_mmmu_dataset
        
        sample_data = {
            "id": ["q1"],
            "question": ["What is 2+2?"],
            "image_1": [None],
            "answer": ["A"]
        }
        
        mock_dataset = Dataset.from_dict(sample_data)
        mock_load_dataset.return_value = mock_dataset
        
        with patch.object(mock_dataset, 'cast_column', return_value=mock_dataset):
            result = load_mmmu_dataset()
        
        # Should default to empty list
        self.assertEqual(result[0]["options"], [])
    
    @patch('data_loader.load_dataset')
    def test_preprocess_all_fields_present(self, mock_load_dataset):
        """Test that all expected fields are present in preprocessed data"""
        from data_loader import load_mmmu_dataset
        
        sample_data = {
            "id": ["q1"],
            "question": ["Test question"],
            "image_1": [None],
            "options": ["['A', 'B']"],
            "answer": ["A"]
        }
        
        mock_dataset = Dataset.from_dict(sample_data)
        mock_load_dataset.return_value = mock_dataset
        
        with patch.object(mock_dataset, 'cast_column', return_value=mock_dataset):
            result = load_mmmu_dataset()
        
        self.assertIn("id", result[0])
        self.assertIn("question", result[0])
        self.assertIn("image", result[0])
        self.assertIn("options", result[0])
        self.assertIn("label", result[0])
    
    @patch('builtins.print')
    @patch('data_loader.load_dataset')
    def test_print_loading_message(self, mock_load_dataset, mock_print):
        """Test that loading message is printed"""
        from data_loader import load_mmmu_dataset
        
        mock_dataset = Mock()
        mock_dataset.cast_column.return_value = mock_dataset
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset
        
        load_mmmu_dataset(split="test", subject="Physics")
        
        mock_print.assert_any_call("Loading MMMU dataset split: test, subject: Physics")
    
    @patch('data_loader.load_dataset')
    def test_different_splits(self, mock_load_dataset):
        """Test loading different dataset splits"""
        from data_loader import load_mmmu_dataset
        
        mock_dataset = Mock()
        mock_dataset.cast_column.return_value = mock_dataset
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset
        
        splits = ["train", "test", "validation"]
        
        for split in splits:
            mock_load_dataset.reset_mock()
            load_mmmu_dataset(split=split)
            call_args = mock_load_dataset.call_args
            self.assertEqual(call_args[1]["split"], split)
    
    @patch('data_loader.load_dataset')
    def test_different_subjects(self, mock_load_dataset):
        """Test loading different subjects"""
        from data_loader import load_mmmu_dataset
        
        mock_dataset = Mock()
        mock_dataset.cast_column.return_value = mock_dataset
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset
        
        subjects = ["Physics", "Chemistry", "Biology"]
        
        for subject in subjects:
            mock_load_dataset.reset_mock()
            load_mmmu_dataset(subject=subject)
            call_args = mock_load_dataset.call_args
            self.assertEqual(call_args[0][1], subject)
    
    @patch('builtins.print')
    @patch('data_loader.load_dataset')
    def test_preprocess_warning_on_parse_failure(self, mock_load_dataset, mock_print):
        """Test that warning is printed when options parsing fails"""
        from data_loader import load_mmmu_dataset
        
        sample_data = {
            "id": ["q1"],
            "question": ["Test"],
            "image_1": [None],
            "options": ["invalid{syntax}"],
            "answer": ["A"]
        }
        
        mock_dataset = Dataset.from_dict(sample_data)
        mock_load_dataset.return_value = mock_dataset
        
        with patch.object(mock_dataset, 'cast_column', return_value=mock_dataset):
            result = load_mmmu_dataset()
        
        warning_printed = any("Warning: Failed to parse options" in str(call_args) 
                              for call_args in mock_print.call_args_list)
        self.assertTrue(warning_printed)
    
    @patch('data_loader.load_dataset')
    def test_multiple_rows_processed_correctly(self, mock_load_dataset):
        """Test that multiple rows are processed correctly"""
        from data_loader import load_mmmu_dataset
        
        sample_data = {
            "id": ["q1", "q2", "q3"],
            "question": ["Q1?", "Q2?", "Q3?"],
            "image_1": [None, None, None],
            "options": ["['A', 'B']", "['X', 'Y']", "['1', '2']"],
            "answer": ["A", "X", "1"]
        }
        
        mock_dataset = Dataset.from_dict(sample_data)
        mock_load_dataset.return_value = mock_dataset
        
        with patch.object(mock_dataset, 'cast_column', return_value=mock_dataset):
            result = load_mmmu_dataset()
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["options"], ['A', 'B'])
        self.assertEqual(result[1]["options"], ['X', 'Y'])
        self.assertEqual(result[2]["options"], ['1', '2'])


if __name__ == '__main__':
    unittest.main()