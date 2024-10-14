#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytesseract
from PIL import Image

# ----------------------- Hardcoded Constants -----------------------

# Input image path
IMAGE_PATH = "xx"  # Write your image path here

# ----------------------- Main Function -----------------------

def main():
    
    # Read the image
    image = Image.open(IMAGE_PATH)

    # Perform OCR
    print("-------> Performing OCR prediction")
    text = pytesseract.image_to_string(image)
    print("-------> OCR prediction completed")

    # Output OCR result
    print("-------> OCR result:")
    print(text)


if __name__ == "__main__":
    main()
