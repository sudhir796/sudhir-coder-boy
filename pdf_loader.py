from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Placeholder text - replace with your actual document content or loaded PDF text
text_content = "This is some example text that we want to split into smaller chunks. LangChain's CharacterTextSplitter can help with this."

splitter = CharacterTextSplitter(
    separator="",  # Explicitly set separator to empty string for character-level splitting
    chunk_size=30,
    chunk_overlap=2,
    length_function=len
)

chunks = splitter.split_text(text_content)
print(chunks)

embedding_model=HuggingFaceEmbeddings()

db=FAISS.from_texts(
    chunks,
    embedding_model
)

result=db.similarity_search("What is LangChain?")
print(result[0].page_content)

pdf_path="/content/Sudhir_resume (3).pdf"
loader=PyPDFLoader(pdf_path)
documents=loader.load()
documents