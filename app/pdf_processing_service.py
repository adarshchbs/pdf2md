import fitz


def process_pdf_stub(file_path: str) -> str:
    """
    This is a stub function to imitate the process of PDF parsing.

    :param file_path: The path to the PDF file.
    :return: A predetermined string for simulation.
    """
    doc = fitz.open(file_path)  # open a document

    return "\n\n".join([page.get_text() for page in doc])
