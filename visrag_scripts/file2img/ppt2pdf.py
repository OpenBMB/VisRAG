import win32com.client

def ppt_to_pdf(input_file, output_file):
    # Create a PowerPoint application instance
    powerpoint = win32com.client.Dispatch("PowerPoint.Application")
    # Set PowerPoint to be visible
    powerpoint.Visible = 1

    # Open the PowerPoint presentation
    presentation = powerpoint.Presentations.Open(input_file, WithWindow=False)

    # Save the presentation as PDF format
    presentation.SaveAs(output_file, 32)  # 32 represents PDF format
    presentation.Close()
    powerpoint.Quit()

# Example usage
input_path = r"C:\path\to\your\file.pptx"
output_path = r"C:\path\to\your\file.pdf"
ppt_to_pdf(input_path, output_path)
