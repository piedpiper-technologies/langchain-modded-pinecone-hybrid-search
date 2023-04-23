# PineconeHybridVectorCreator and Modified HybridSearchRetriever Module Files for [LangChain](https://github.com/hwchase17/langchain)

This README provides an overview of a custom module **PineconeHybridVectorCreator** and the **modified PineconeHybridSearchRetriever** for Langchain. 

These tools offer several advantages over the previous version of the original Hybrid Search Retriever, enhancing the generation of hybrid sparse-dense vectors from text inputs and their retrieval from a Pinecone.io hybrid index.  

To use these custom files in your project download a copy of the `pinecone_hybrid_vector_creator.py` and add it to your project.  For the **PineconeHybridSearchRetriever** copy the raw text from the repo and paste the contents in your `./langchain/langchain/retrievers/pinecone_hybrid_search.py` file.  The `pinecone_hybrid_search.py` must be modified if you intend to use the **PineconeHybridVectorCreator** class function.

This repo will be linked to a disussion on [LangChain](https://github.com/hwchase17/langchain) Discussions for consideration as part of the offical modules if you wish to contribute to the conversation. 

## Key Benefits

1. **Flexibility**: The Hybrid Vector Creator generates hybrid vectors separately from the search process, providing more flexibility in managing indexing and retrieval processes and incorporating generated hybrid vectors into other workflows.
2. **Optimization**: The modified Hybrid Search Retriever optimizes the alpha parameter used for scaling dense and sparse vectors, leading to improved search results by more effectively determining the optimal balance between dense and sparse vector contributions.
3. **Metadata handling**: The Hybrid Vector Creator supports adding source information to the metadata of each vector, which is useful for tracking and filtering search results using the updated retriever, as well as incorporating into chains that require source information. The previous version of the Hybrid Search Retriever did not offer this functionality.
4. **Namespace Declaration**: Both the updated Hybrid Search Retriever and the Hybrid Vector Creator can pass a namespace argument for vector creation and retrieval. This enables developers to create richer indexing and retrieval options, allowing for more secure data management by limiting search to vectors in a namespace tied to user roles or other variables.
5. **Modularity**: The Hybrid Vector Creator and the modified Hybrid Search Retriever are designed for easy integration into different workflows, streamlining processes and making it easier to adapt and extend functionalities to fit specific use cases.
6. **Readability**: The code for both the Hybrid Vector Creator and the modified Hybrid Search Retriever is more readable and easier to understand compared to the previous version, facilitating easier maintenance and future development.
7. **Developer Friendly**: The Hybrid Vector Creator returns a sparse-dense vector list of dictionaries for upsert rather than performing the upsert in the module, providing developers with better control and visualization options in their scripts.

## Example Using PineconeHybridVectorCreator

The following example demonstrates how to:

1. Load input texts from a PDF file and split them using a custom text splitter.
2. Prepare input texts and metadata.
3. Load LlamaCpp embeddings and BM25 sparse encoder.
4. Create Pinecone hybrid vectors using the PineconeHybridVectorCreator.
5. Upsert the hybrid vectors into a Pinecone index.

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone_text.sparse import BM25Encoder
from langchain.embeddings import LlamaCppEmbeddings
from pinecone_hybrid_vector_creator import PineconeHybridVectorCreator
import pinecone
import os

# Initialize Pinecone
pinecone.init(api_key="your-pinecone-api-key", environment="your-pinecone-environment")

# Initialize Pinecone index
index_name = "your-index-name"
index = pinecone.Index(index_name)

# Load and split input texts from a PDF file
pdf_path = "path/to/your/pdf"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20, length_function=len)
loader = PyPDFLoader(pdf_path)
texts = loader.load_and_split(text_splitter=text_splitter)

# Extract file name from PDF path
file_name = pdf_path.split("/")[-1]

# Prepare input texts and metadata
corpus = []
metadata = []
for text in texts:
    context = text.page_content.replace("\n", " ")
    corpus.append(context)
    meta = {"source": file_name}
    metadata.append(meta)

# Load LlamaCpp embeddings
llama_path = "path/to/your/llama-cpp/embeddings"
embeddings = LlamaCppEmbeddings(model_path=llama_path)

# Fit and dump BM25 sparse encoder
encoder = BM25Encoder().default()
encoder.fit(corpus)
encoder.dump("bm25_values.json")

# Load BM25 sparse encoder from JSON file
encoder = BM25Encoder().load("bm25_values.json")

# Create Pinecone hybrid vectors
vector_creator = PineconeHybridVectorCreator(embeddings=embeddings, sparse_encoder=encoder)
sparse_dense = vector_creator.generate_vectors(contexts=corpus, meta_dicts=metadata)

# Upsert vectors to Pinecone index
index.upsert(sparse_dense)

# Delete BM25 values file
os.remove("bm25_values.json")

```

## Example Using Modified PineconeHybridSearchRetriever

The following example demonstrates how to:

1. Load LlamaCpp embeddings and BM25 sparse encoder.
2. Initialize Pinecone and create a Pinecone index.
3. Set up a PineconeHybridSearchRetriever with the embeddings and sparse encoder.
4. Perform a search query and retrieve relevant documents.
5. Process the search results and generate a response using the ChatOpenAI model.


```python
from langchain.retrievers import PineconeHybridSearchRetriever
from langchain.embeddings import LlamaCppEmbeddings
from pinecone_text.sparse import BM25Encoder
import pinecone, os

local_path = 'path/to/your/llama-cpp/embeddings'
embeddings = LlamaCppEmbeddings(model_path=local_path)
sparse_encoder = BM25Encoder().default()
corpus = []
query = "your-query-text"
corpus.append(query)
sparse_encoder.fit(corpus)

sparse_encoder.dump("bm25_values.json")

sparse_encoder = BM25Encoder().load("bm25_values.json")

pinecone.init(api_key="your-pinecone-api-key", environment="your-pinecone-environment")

index_name = "your-index-name"
index = pinecone.Index(index_name)

retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=sparse_encoder, index=index)

result = retriever.get_relevant_documents(query)

result_chunk = ""

for document in result:
    page_content = document.page_content.replace('\n', ' ')
    result_chunk += page_content

# Delete BM25 values file
os.remove("bm25_values.json")

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
chat = ChatOpenAI(temperature=0,streaming=True)
messages = [
    SystemMessage(content="You are a helpful assistant that answers a 'USER_QUESTION:' by summarizing the 'CONTEXT:' that comes with the question.  If an answer cannot be formed using context reply 'I dont know'."),
    HumanMessage(content="USER_QUESTION: " + query + " CONTEXT: " + result_chunk)
]
chat(messages)

```
