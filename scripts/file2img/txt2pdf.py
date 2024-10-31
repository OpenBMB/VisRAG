from fpdf import FPDF

def txt_to_pdf(input_file, output_file):
    # Create a PDF object
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Set font for the PDF document
    pdf.set_font("Arial", size=12)

    # Read the text file line by line
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            # Add a line to the PDF
            pdf.cell(200, 10, txt=line, ln=True)

    # Save the output PDF
    pdf.output(output_file)

# Example usage
input_path = r"C:\path\to\your\file.txt"
output_path = r"C:\path\to\your\file.pdf"
txt_to_pdf(input_path, output_path)
