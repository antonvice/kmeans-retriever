## Loading + Parsing
from bs4 import BeautifulSoup
import requests
from typing import List

def fetch_html_content(url: str) -> str:
    response = requests.get(url)
    return response.text

def parse_html_to_paragraphs(html_content: str) -> List[str]:
    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = [p.get_text() for p in soup.find_all('p')]
    return paragraphs

## encoding + elbow
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import numpy as np

def encode_documents(model, documents: List[str]) -> np.ndarray:
    return model.encode(documents)

def calculate_inertias(embeddings: np.ndarray, k_range: range) -> List[float]:
    return [KMeans(n_clusters=k, random_state=42).fit(embeddings).inertia_ for k in k_range]

def find_elbow(inertias: List[float]) -> int:
    n_points = len(inertias)
    all_coords = np.vstack((range(n_points), inertias)).T
    line_vec = all_coords[-1] - all_coords[0]
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    vec_from_first = all_coords - all_coords[0]
    scalar_product = np.dot(vec_from_first, line_vec_norm)
    vec_to_line = vec_from_first - np.outer(scalar_product, line_vec_norm)
    dist_to_line = np.linalg.norm(vec_to_line, axis=1)
    return np.argmax(dist_to_line)

## Clustering + Retrieving

def cluster_and_retrieve_docs(embeddings: np.ndarray, query_index: int, num_docs: int) -> List[int]:
    elbow_index = find_elbow(calculate_inertias(embeddings, range(1, 10)))
    kmeans = KMeans(n_clusters=elbow_index, random_state=42).fit(embeddings)
    query_cluster = kmeans.labels_[query_index]
    cluster_indices = [i for i, cluster_id in enumerate(kmeans.labels_) if cluster_id == query_cluster and i != query_index]
    return np.random.choice(cluster_indices, size=min(num_docs, len(cluster_indices)), replace=False) if len(cluster_indices) > num_docs else cluster_indices


## Main
import argparse

def main(url: str, query: str, num_docs: int):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    html_content = fetch_html_content(url)
    documents = parse_html_to_paragraphs(html_content)
    all_documents = documents + [query]
    embeddings = encode_documents(model, all_documents)
    selected_indices = cluster_and_retrieve_docs(embeddings, len(documents), num_docs)
    for index in selected_indices:
        print(documents[index])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Document Retrieval with KMeans Clustering')
    parser.add_argument('-i', '--input-url', required=True, help='URL of the document to process')
    parser.add_argument('-q', '--query', required=True, help='Query to match in the document')
    parser.add_argument('-k', '--num-docs', type=int, required=True, help='Number of documents to retrieve')
    args = parser.parse_args()
    
    main(args.input_url, args.query, args.num_docs)
