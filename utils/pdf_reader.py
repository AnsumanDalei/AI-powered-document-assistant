import fitz
import os


def extract_pdf_text(pdf_path: str) -> str:
    """Extract selectable text from each page of a PDF file."""
    if os.path.getsize(pdf_path) == 0:
        raise ValueError("This PDF is empty and contains no data.")
    try:
        with fitz.open(pdf_path) as document:
            if document.needs_pass:
                raise ValueError("This PDF is password-protected.")
            if len(document) == 0:
                raise ValueError("This PDF is empty and contains no pages.")
            text_parts = [page.get_text() for page in document]
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("This file is invalid or could not be opened as a PDF.") from error
    extracted_text = "\n".join(text_parts)
    if not extracted_text.strip():
        raise ValueError("No selectable text was found. This may be a scanned or image-only PDF; OCR is not included.")
    return extracted_text
