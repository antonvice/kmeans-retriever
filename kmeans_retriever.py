## Loading + Parsing
from bs4 import BeautifulSoup
import requests
from typing import List
import matplotlib.pyplot as plt


def fetch_html_content(url: str) -> str:
    response = requests.get(url)
    return response.text

def parse_html_to_paragraphs(html_content: str) -> List[str]:
    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = [p.get_text() for p in soup.find_all('p')]
    return paragraphs

## encoding + elbow
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

def encode_documents(model, documents: List[str]) -> np.ndarray:
    return model.encode(documents)

def calculate_inertias(embeddings: np.ndarray, k_range: range) -> List[float]:
    return [KMeans(n_clusters=k, random_state=42).fit(embeddings).inertia_ for k in k_range]

def plot_elbow_curve(k_range: range, inertias: List[float]):
        # Plotting the elbow graph
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, '-o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.xticks(k_range)
    plt.grid(True)

    # Annotating each point with its k value for clarity
    for i, inertia in enumerate(inertias):
        plt.annotate(f'k={i+1}', (k_range[i], inertias[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.show()
def find_elbow(inertias: List[float]) -> int:
    n_points = len(inertias)
    all_coords = np.vstack((range(n_points), inertias)).T
    line_vec = all_coords[-1] - all_coords[0]
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    vec_from_first = all_coords - all_coords[0]
    scalar_product = np.dot(vec_from_first, line_vec_norm)
    vec_to_line = vec_from_first - np.outer(scalar_product, line_vec_norm)
    dist_to_line = np.linalg.norm(vec_to_line, axis=1)
    elbow_index = np.argmax(dist_to_line)
    print(f"Elbow index: {elbow_index}")
    return elbow_index

## Clustering + Retrieving

def cluster_documents(embeddings: np.ndarray) -> KMeans:
    k_range = range(1, 11)
    inertias = calculate_inertias(embeddings, k_range)
    elbow_index = find_elbow(inertias)
    kmeans = KMeans(n_clusters=elbow_index, random_state=42).fit(embeddings)
    return kmeans

def retrieve_docs_from_query_cluster(kmeans: KMeans, query_cluster:int, num_docs: int) -> List[int]:
    print("Query cluster: ", query_cluster)
    cluster_indices = [i for i, cluster_id in enumerate(kmeans.labels_) if cluster_id == query_cluster]
    print("Cluster indices: ", cluster_indices)
    return np.random.choice(cluster_indices, size=min(num_docs, len(cluster_indices)), replace=False) if len(cluster_indices) > num_docs else cluster_indices

def compute_cosine_similarity(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    return np.dot(query_embedding, embeddings.T) / (np.linalg.norm(query_embedding) * np.linalg.norm(embeddings, axis=1))

def retrieve_similar_docs(query_embedding: np.ndarray, embeddings: np.ndarray, num_docs: int, documents: List[str]) -> List[str]:
    # Compute cosine similarity between the query embedding and each document embedding
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings).flatten()

    # Sort the documents based on their similarity to the query, in descending order
    sorted_indices = np.argsort(similarities)[::-1]

    # Select the top `num_docs` similar documents, ensuring we do not go out of bounds
    selected_documents = [documents[i] for i in sorted_indices[:num_docs]]

    return selected_documents
def retrieve_similar_docs_from_cluster(query_embedding: np.ndarray, embeddings: np.ndarray, num_docs: int, documents: List[str], query_cluster: int, kmeans: KMeans) -> List[str]:
    # Filter embeddings and documents to only include those in the same cluster as the query
    cluster_indices = [i for i, cluster_id in enumerate(kmeans.labels_) if cluster_id == query_cluster]
    cluster_embeddings = embeddings[cluster_indices]
    cluster_documents = [documents[i] for i in cluster_indices]

    # Compute cosine similarity between the query embedding and each document embedding in the cluster
    similarities = cosine_similarity(query_embedding.reshape(1, -1), cluster_embeddings).flatten()

    # Sort the documents based on their similarity to the query, in descending order
    sorted_indices = np.argsort(similarities)[::-1]

    # Select the top `num_docs` similar documents
    selected_documents = [cluster_documents[i] for i in sorted_indices[:num_docs]]

    return selected_documents

## Main
import argparse

def main(url: str, query: str, num_docs: int):
    # Load the Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Fetch HTML content and parse it into paragraphs
    html_content = fetch_html_content(url)
    documents = parse_html_to_paragraphs(html_content)
    
    # Append the query as the last document for clustering
    all_documents = documents + [query]
    
    # Encode all documents, including the query
    embeddings = encode_documents(model, all_documents)
    
    # Perform clustering on the embeddings
    kmeans = cluster_documents(embeddings)
    
    # The query index is the last one in the embeddings array
    query_cluster = kmeans.labels_[-1]
    
    # Retrieve indices of documents from the query's cluster
    selected_indices = retrieve_docs_from_query_cluster(kmeans, query_cluster, num_docs)
    
    # Print the selected documents
    for index in selected_indices:
       print(documents[index])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Document Retrieval with KMeans Clustering')
    
    # Define a test flag
    parser.add_argument('-t', '--test', action='store_true', help='use test args')
    
    # Define the other arguments without making them strictly required
    parser.add_argument('-i', '--input-url', help='URL of the document to process')
    parser.add_argument('-q', '--query', help='Query to match in the document')
    parser.add_argument('-k', '--num-docs', type=int, help='Number of documents to retrieve')

    args = parser.parse_args()

    # If test flag is used, run the test scenario
    if args.test:
        main("https://lilianweng.github.io/posts/2023-06-23-agent/", "Chain of Thought", 3)
    else:
        # Ensure the other arguments are provided when not in test mode
        if not args.input_url or not args.query or args.num_docs is None:
            parser.error("The arguments -i, -q, and -k are required unless -t is specified.")
        else:
            main(args.input_url, args.query, args.num_docs)