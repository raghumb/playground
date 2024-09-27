from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
 
def load_documents():
     loader = DirectoryLoader('../../files', glob="**/*.txt")
     documents = loader.load()
     print("number of documents " + str(len(documents)))
     return documents
 
def split_text(documents: list[Document]):
     text_splitter = RecursiveCharacterTextSplitter(
         chunk_size=1000, # Size of each chunk in characters
         chunk_overlap=200, # Overlap between consecutive chunks
         length_function=len, # Function to compute the length of the text
         add_start_index=True, # Flag to add start index to each chunk
       )
     
     chunks = text_splitter.split_documents(documents)
     return chunks
 
def save_to_db(chunks: list[Document], vector_store):
 
     vector_store.add_documents(documents=chunks)
     print(f"Saved {len(chunks)} chunks ")
 
def ingest_data(vector_store):
     documents = load_documents() # Load documents from a source
     chunks = split_text(documents) # Split documents into manageable chunks
     save_to_db(chunks, vector_store) # Save the processed data to a data store