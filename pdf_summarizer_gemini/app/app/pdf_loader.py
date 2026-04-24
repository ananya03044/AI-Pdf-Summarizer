from pypdf import PdfReader


def load_pdf(path: str) -> str:
    """
    Loads a PDF file and extracts all text from it.
    """
    reader = PdfReader(path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text
