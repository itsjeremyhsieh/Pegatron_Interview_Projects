import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
from model_interface import BenchmarkModel


class TestBenchmarkModel(unittest.TestCase):

    def setUp(self):
        self.img = Image.new("RGB", (10, 10), color="red")
        self.model = BenchmarkModel(model_name="gpt-4o-mini")
        self.model = BenchmarkModel("test-model")
        # Create a simple 1x1 pixel image for testing
        self.test_image = Image.new("RGB", (1, 1), color="white")
        self.question = "What color is the pixel?"
        self.options = ["Red", "Green", "White", "Blue"]

    def test_pil_to_data_url(self):
        data_url = self.model._pil_to_data_url(self.img)
        self.assertTrue(data_url.startswith("data:image/jpeg;base64,"))
        self.assertIn("base64", data_url)

        
    @patch("model_interface.OpenAI")
    def test_predict_returns_mocked_response(self, mock_openai_class):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="C"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        model = BenchmarkModel("test-model")
        result = model.predict(self.question, self.test_image, self.options)
        self.assertEqual(result, "C")
        mock_client.chat.completions.create.assert_called_once()


    @patch("model_interface.OpenAI")
    def test_predict_handles_exception(self, mock_openai_class):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API failure")
        mock_openai_class.return_value = mock_client

        result = self.model.predict(self.question, self.test_image, self.options)
        self.assertEqual(result, "")  

    def test_pil_to_data_url(self):
        data_url = self.model._pil_to_data_url(self.test_image)
        self.assertTrue(data_url.startswith("data:image/jpeg;base64,"))
        
if __name__ == "__main__":
    unittest.main()
