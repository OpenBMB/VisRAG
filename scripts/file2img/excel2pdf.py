import win32com.client

def excel_to_pdf(input_file, output_file):
    # Create an Excel application instance
    excel = win32com.client.Dispatch("Excel.Application")
    # Set Excel to be invisible
    excel.Visible = False

    # Open the Excel workbook
    workbook = excel.Workbooks.Open(input_file)

    # Export the workbook as PDF format
    workbook.ExportAsFixedFormat(0, output_file)  # 0 represents PDF format
    workbook.Close()
    excel.Quit()

# Example usage
input_path = r"C:\path\to\your\file.xlsx"
output_path = r"C:\path\to\your\file.pdf"
excel_to_pdf(input_path, output_path)
