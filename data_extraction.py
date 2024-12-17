import fitz  # PyMuPDF
import os
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

def PyMuPDF_loader(directory_path):
    """
    Reads a directory of PDF files, extracts text from each file, and returns a list of extracted text.
    
    Args:
        directory_path (str): Path to the directory containing PDF files.
    
    Returns:
        list: List of extracted text from PDF files in the directory.
    """
    pdf_files = []
    extracted_text = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_files.append(os.path.join(directory_path, filename))
    
    for pdf_file in pdf_files:
        text_content = []
        with fitz.open(pdf_file) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_content.append(page.get_text("text"))  # Extracts text from each page         
        extracted_text.append("\n".join(text_content))
    
    return extracted_text
    

def pypdf_loader(directory_path):

    loader = DirectoryLoader(
    directory_path, glob="./*/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    return documents

