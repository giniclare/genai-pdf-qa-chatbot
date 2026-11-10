## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:

To develop a chatbot that can read and understand PDF documents and answer user questions based on the information available in the document. The system uses LangChain to extract text, split it into smaller chunks, generate embeddings, and retrieve relevant information to provide accurate answers.

### DESIGN STEPS:

#### STEP 1: Load and Extract PDF Content
- Load the PDF document using PyPDFLoader.
- Extract text from all pages of the PDF.
- Store the extracted content for further processing.
#### STEP 2: Text Processing and Vector Database Creation
- Split the extracted text into smaller chunks using RecursiveCharacterTextSplitter.
- Generate embeddings using OpenAIEmbeddings.
- Store the embeddings in a vector database using DocArrayInMemorySearch.
#### STEP 3: Create Chatbot and Generate Answers
- Create a question-answering chain using RetrievalQA.
- Use ChatOpenAI to generate answers.
- Accept questions from the user and retrieve relevant information from the PDF.
- Display the generated answers until the user enters exit.


### PROGRAM:
```python
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load API Key
load_dotenv()

# Load PDF
loader = PyPDFLoader("AI_Notes.pdf")
documents = loader.load()

# Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Create Vector Database
embeddings = OpenAIEmbeddings()
db = DocArrayInMemorySearch.from_documents(docs, embeddings)

# Create QA Chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    retriever=db.as_retriever()
)

# Ask Questions
while True:
    question = input("You: ")

    if question.lower() == "exit":
        break

    answer = qa.run(question)
    print("Bot:", answer)

```

### OUTPUT:

<img width="1401" height="563" alt="image" src="https://github.com/user-attachments/assets/7d2d508a-6d41-4f96-8fcc-f509a49993f9" />

### RESULT:

Thus, a PDF-based question-answering chatbot was successfully developed using LangChain. The chatbot loads PDF documents, extracts and processes their content, and answers user queries by retrieving relevant information from the document.
