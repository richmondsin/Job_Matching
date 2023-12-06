# For windows users, please remove | tail -n 1 below
# pip install pymilvus | tail -n 1
# pip install sentence_transformers | tail -n 1
# pip install pandas | tail -n 1
# pip install rouge_score | tail -n 1
# pip install nltk | tail -n 1
# pip install "ibm-watson-machine-learning>=1.0.312" | tail -n 1
# pip install --upgrade ibm_watson_machine_learning

import os
import re
import pandas as pd
from typing import Optional, Dict, Any, Iterable, List
import pdb
from langchain.document_loaders import PyPDFLoader
import numpy as np

try: 
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Could not import sentence_transformers. Please install sentence-transformers package.")

try:
    from pymilvus import Milvus, DataType, connections
except ImportError:
    raise ImportError("Could not import Milvus: Please install Milvus package.")

# Index knowledge base
# Load data
datasets = ['nuhs']
dataset = datasets[0]    # The current dataset to use
data_root = "data"
data_dir = os.path.join(data_root, dataset)
max_docs = -1
# print("Selected dataset:", dataset)

def load_data_v1(data_dir, data_root):
    passages = pd.read_csv(os.path.join(data_dir, "jobs.tsv"), sep='\t', header=0)
    # We only use 5000 examples.  Comment the lines below to use the full dataset.
    passages = passages.head(5000)
    return passages

documents = load_data_v1(data_dir, data_root)
documents['indextext'] = documents['id'].astype(str) + "\n" + documents['job_description'] + "\n" + documents['job']

# Create embedding function
class MiniLML6V2EmbeddingFunction():
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    def __call__(self, texts):
        return MiniLML6V2EmbeddingFunction.MODEL.encode(texts).tolist()
emb_func = MiniLML6V2EmbeddingFunction()

# Create a Milvus collection for embeddings
milvus = Milvus(host='localhost', port='19530')
collection_name = "document_embeddings"
dimension = 768  # Change this dimension to match your embedding dimension

if not milvus.has_collection(collection_name):
    milvus.create_collection(
        collection_name,
        fields=[
            {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": dimension}}
        ],
    )

# Load a pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Insert document embeddings into Milvus collection
embeddings = model.encode(documents.indextext.tolist())
ids = [str(i) for i in documents.id]

status, ids = milvus.insert(
    collection_name=collection_name,
    records={"embedding": np.array(embeddings)},
    ids=ids
)

# Load the pdf file
def pdf_to_text(path: str,
                start_page: int = 1,
                end_page: Optional[int | None] = None) -> list[str]:
    """
    Convert PDF to plain text.

    Params:
    path (str): Path to PDF file.
    start_page (int): Page to start getting the text from.
    end_page (int): Last page to get text from.
    """
    loader = PyPDFLoader(path)
    pages = loader.load()
    total_pages = len(pages)

    if end_page is None:
        end_page = len(pages)
    
    text_list = []
    for i in range(start_page-1, end_page):
        text = pages[i].page_content
        text = text.replace("\n", " ")  
        text = re.sub(r'\s+', ' ', text)
        text_list.append(text)

    return text_list

resume = pdf_to_text("job_1.pdf")
# print(text_list)

# Search for relevant documents
resume_text = resume[0]
query_vector = model.encode([resume_text])[0]

search_param = {
    "collection_name": collection_name,
    "query_records": [query_vector],
    "top_k": 3,
    "params": {"metric_type": "L2"}
}

results = milvus.search(**search_param)
similar_document_ids = results.results[0].ids

# Retrieve and display the relevant documents
relevant_chunks = [documents[documents['id'] == int(id)] for id in similar_document_ids]
for doc in relevant_chunks:
    print("=========")
    print("Paragraph index : ", doc['id'].values[0])
    print("Paragraph : ", doc['indextext'].values[0])
    
# Select a question
question_text = "What is the most suitable job position for this resume?"
# print(question_text)

# Feed the context and the question to openai model
def make_prompt(context, resume, question_text):
    return (f"Contexts: {context}\n Please answer the question using the contexts above.\n"
            + f"Resume: {resume}\n."
            + f"Question: {question_text}\n"
            + "Job Position: ")


context = "\n\n\n".join(relevant_chunks["documents"][0])
prompt = make_prompt(context, resume[0], question_text)
pdb.set_trace()
# Foundation Models on watsonx
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.foundation_models import Model

# watsonx API connection
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "pB4tWiZhs2a5fD2_zuXtZBGmGrPGlvtiVyHFOOPfg0xe"
}

# project id
try:
    project_id = "c6f504c9-cee0-4f48-a408-a032ee344055"
except KeyError:
    project_id = input("Please enter your project_id (hit enter): ")

model_id = ModelTypes.LLAMA_2_70B_CHAT

# # Greedy method
# parameters = {
#     GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
#     GenParams.MIN_NEW_TOKENS: 1,
#     GenParams.MAX_NEW_TOKENS: 100,
# }

# Sample method
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.TOP_K: 20,
    GenParams.TOP_P: 0.8,
    GenParams.TEMPERATURE: 0.8,
    GenParams.MAX_NEW_TOKENS: 20
}

model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

# Generate the answers
results = model.generate_text(prompt=prompt)
print(results)