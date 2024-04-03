import unittest
from handlers.handlers import img2txt

class TestHandlers(unittest.TestCase):
    def test_img2txt(self):
        input_text = "Describe the image"
        input_image = "sample_image.jpg"
        description = img2txt(input_text, input_image)
        self.assertIsInstance(description, str)
        self.assertTrue(len(description) > 0)

if __name__ == '__main__':
    unittest.main()
