import pandas as pd
from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber
import io

def parse_tabular_data(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """Parses CSV or Excel into a pandas DataFrame."""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            raise ValueError("Unsupported tabular format")
        return df
    except Exception as e:
        raise ValueError(f"Error parsing tabular data: {e}")

def parse_pdf(file_bytes: bytes) -> List[Document]:
    """Parses PDF into a list of LangChain Documents."""
    documents = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"page": i + 1}
                        )
                    )
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return splitter.split_documents(documents)
    except Exception as e:
        raise ValueError(f"Error parsing PDF data: {e}")
