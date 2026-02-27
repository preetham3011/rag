"""PDF text extraction using PyMuPDF (fitz)"""

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str) -> list:
    """
    Extract text from PDF file page by page.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of dictionaries with format:
        [
            {"page": 1, "text": "page 1 content..."},
            {"page": 2, "text": "page 2 content..."},
            ...
        ]
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If PDF cannot be opened or read
    """
    pages_data = []
    
    try:
        # Open the PDF document
        doc = fitz.open(pdf_path)
        
        # Extract text from each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Skip empty pages
            if text.strip():
                pages_data.append({
                    "page": page_num + 1,  # 1-indexed page numbers
                    "text": text
                })
        
        # Close the document
        doc.close()
        
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    return pages_data


if __name__ == "__main__":
    # Test the PDF extractor
    import sys
    
    # Example usage
    test_pdf_path = "data/raw/sample.pdf"
    
    print(f"Testing PDF extraction on: {test_pdf_path}")
    print("-" * 50)
    
    try:
        pages = extract_text_from_pdf(test_pdf_path)
        
        print(f"Successfully extracted {len(pages)} pages\n")
        
        # Display first 2 pages as preview
        for page_data in pages[:2]:
            print(f"Page {page_data['page']}:")
            print(page_data['text'][:300] + "..." if len(page_data['text']) > 300 else page_data['text'])
            print("-" * 50)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo test, place a PDF file at 'data/raw/sample.pdf'")
    except Exception as e:
        print(f"Error: {e}")
