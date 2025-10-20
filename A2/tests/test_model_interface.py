import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
from model_interface import BenchmarkModel  

class TestBenchmarkModel(unittest.TestCase):
    
    def setUp(self):
        # Create a simple image for testing
        self.img = Image.new("RGB", (10, 10), color="red")
        self.model = BenchmarkModel(model_name="gpt-4o-mini")
    
    def test_pil_to_data_url(self):
        # Test that the image is correctly converted to a data URL
        data_url = self.model._pil_to_data_url(self.img)
        self.assertTrue(data_url.startswith("data:image/jpeg;base64,"))
        self.assertIn("base64", data_url)

if __name__ == "__main__":
    unittest.main()
