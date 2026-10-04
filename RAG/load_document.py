from langchain_community.document_loaders import PyPDFLoader, TextLoad
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load your local document 
# loader = TextLoader("sample.txt") #Or PyPDFLoader("manual")
pdf_files = [
    "Statistics DS notes.pdf" , "mdm file.pdf" , "gsm architecture.pdf"
]
documents = []
for pdf_file in pdf_files:
    documents.extend(PyPDFLoader(pdf_file).load())

# 2. Split document into overlapping chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap = 50
)
chunks = text_splitter.split_documents(documents)
print(f"Total document chunks created: {len(chunks)}")
