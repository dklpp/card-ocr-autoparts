import cv2
from PIL import Image

import pytesseract

from pytesseract import Output

#pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

im_file = "data/img10.jpg"


def image_to_text(input_path):
   """
   A function to read text from images.
   """
   img = cv2.imread(input_path)
   text = pytesseract.image_to_string(img)

   return text.strip()


print(image_to_text(im_file))


data = pytesseract.image_to_data(im_file, output_type=Output.DICT)

print(data.keys())
