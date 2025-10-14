from pdf2image import convert_from_path

def pdf_to_images(pdf_path, output_folder):
    # Convert PDF to list of PIL Image objects
    images = convert_from_path(pdf_path, dpi=300)
    
    # Save each image as a separate file
    for i, image in enumerate(images):
        image_path = f"{output_folder}/page_{i + 1}.png"
        image.save(image_path, 'PNG')

# Example usage
pdf_to_images("path/to/your/input.pdf", "path/to/your/output/folder")
