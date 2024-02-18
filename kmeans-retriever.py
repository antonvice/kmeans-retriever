# Implementation without Langchain

#### LOADER

#* This is just a web based loader, 
#* need to implement other loaders


from bs4 import BeautifulSoup
import requests
from typing import List

def parse_page(url:str)-> List[str]:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = [p.get_text() for p in soup.find_all('p')] 
    return paragraphs

pages = parse_page('https://lilianweng.github.io/posts/2023-06-23-agent/')

## TEXT SPLITTER
'''

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)

texts = text_splitter.split_documents(pages)


'''
## HERE I WILL IMPLEMENT CLUSTERING with kmeans
# add query to docs, embed all, kmeans, retrieve k of elements

from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import numpy as np
# Load the Sentence Transformer model

from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import numpy as np

# Function to encode documents
def encode_documents(model, documents: List[str]) -> np.array:
    embeddings = model.encode(documents)
    return embeddings

# Function to find the elbow point
def find_elbow(inertias: List[float]) -> int:
    n_points = len(inertias)
    all_coords = np.vstack((range(n_points), inertias)).T
    first_point = all_coords[0]
    last_point = all_coords[-1]
    line_vec = last_point - first_point
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = all_coords - first_point
    scalar_product = np.sum(vec_from_first * line_vec_norm, axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    idx_of_elbow = np.argmax(dist_to_line)
    return idx_of_elbow

# Function to cluster documents and return documents from the query's cluster
def cluster_and_retrieve_docs(model, documents: List[str], query: str, num_docs: int) -> List[str]:
    all_documents = documents + [query]
    embeddings = encode_documents(model, all_documents)
    
    # Perform KMeans clustering
    inertias = []
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=42).fit(embeddings)
        inertias.append(kmeanModel.inertia_)
    
    elbow_index = find_elbow(inertias)
    kmeans = KMeans(n_clusters=elbow_index, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # Identify the cluster that contains the query
    query_cluster = clusters[-1]  # The query is the last document
    
    # Filter documents in the same cluster as the query
    cluster_documents_indices = [i for i, cluster_id in enumerate(clusters) if cluster_id == query_cluster and i != len(clusters) - 1]  # Exclude the query itself
    
    # If there are more documents in the cluster than requested, randomly select 'num_docs'
    if len(cluster_documents_indices) > num_docs:
        selected_indices = np.random.choice(cluster_documents_indices, size=num_docs, replace=False)
    else:
        selected_indices = cluster_documents_indices
    
    # Return the selected documents
    return [documents[i] for i in selected_indices]

# Example usage
model = SentenceTransformer('all-MiniLM-L6-v2')
pages = pages
query = "what are agents?"
num_docs = 3  # Number of documents to return from the cluster

selected_docs = cluster_and_retrieve_docs(model, pages, query, num_docs)
print("Selected documents from the query's cluster:", selected_docs)
