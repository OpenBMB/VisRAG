import win32com.client

def word_to_pdf(input_file, output_file):
    # Create a Word application instance
    word = win32com.client.Dispatch("Word.Application")
    # Set Word to be invisible
    word.Visible = False

    # Open the Word document
    doc = word.Documents.Open(input_file)

    # Save the document as PDF format
    doc.SaveAs(output_file, FileFormat=17)  # FileFormat=17 represents PDF
    doc.Close()
    word.Quit()

# Example usage
input_path = r"C:\path\to\your\file.docx"
output_path = r"C:\path\to\your\file.pdf"
word_to_pdf(input_path, output_path)
