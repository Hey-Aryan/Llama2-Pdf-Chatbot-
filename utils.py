from langchain_community.llms import Ollama
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
import textwrap
import fitz  
from PyPDF2 import PdfReader

def get_pdf_text(pdf_file):
    # Open the PDF file using PyMuPDF (fitz)
    with fitz.open(pdf_file) as doc:
        # Extract text from each page
        text = [page.get_text() for page in doc]
    
    return text

def initialize_embeddings():
    embeddings = OllamaEmbeddings()
    return embeddings

def initialize_llm():
    llm = Ollama(model="llama2")
    return llm

def get_chroma_vectors_db(pages):
    embeddings = initialize_embeddings()
    vector_db = Chroma.from_texts(pages, embeddings)
    return vector_db

def get_question_answer_chain(vector_db, llm):
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type="stuff",
                                           retriever=retriever,
                                           return_source_documents=False)
    return qa_chain

def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def process_llm_response(llm_response):
    return wrap_text_preserve_newlines(llm_response['result'])


def get_pdf_pages(pdf_file):
    # Use PdfReader to open the PDF file
    pdf_reader = PdfReader(pdf_file)

    # Check if the PDF file is encrypted
    if pdf_reader.is_encrypted:
        raise ValueError("The uploaded PDF file is encrypted and cannot be processed.")

    # Extract text from each page
    pages = []
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        pages.append(text)

    return pages

