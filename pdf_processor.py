import fitz

def extract_pages(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        pages = []

        for page_number in range(len(doc)):
            text = doc[page_number].get_text()
            pages.append({
                "page": page_number + 1,
                "text": text
            })
        
        doc.close()
        return pages
    
    except Exception as e:
        raise Exception(f"Failed to extract pages from PDF: {str(e)}")