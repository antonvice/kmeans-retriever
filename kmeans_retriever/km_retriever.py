from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class KMR:
    def __init__(self, url: str):
        self.url = url
        self.documents = self.fetch_and_parse(url)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.encode_documents(self.documents)
        self.kmeans = self.cluster_documents(self.embeddings)
    
    @staticmethod
    def fetch_and_parse(url: str) -> list:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        return paragraphs

    def encode_documents(self, documents: list) -> np.ndarray:
        return self.model.encode(documents)
    
    def cluster_documents(self, embeddings: np.ndarray) -> KMeans:
        k_range = range(1, 11)
        inertias = [KMeans(n_clusters=k, random_state=42).fit(embeddings).inertia_ for k in k_range]
        elbow_index = self.find_elbow(inertias) + 1 # Adjusting for zero indexing
        kmeans = KMeans(n_clusters=elbow_index, random_state=42).fit(embeddings)
        return kmeans
    
    @staticmethod
    def find_elbow(inertias: list) -> int:
        n_points = len(inertias)
        all_coords = np.vstack((range(n_points), inertias)).T
        first_point = all_coords[0]
        line_vec = all_coords[-1] - first_point
        line_vec_norm = line_vec / np.linalg.norm(line_vec)
        vec_from_first = all_coords - first_point
        scalar_product = np.dot(vec_from_first, line_vec_norm)
        vec_to_line = vec_from_first - np.outer(scalar_product, line_vec_norm)
        dist_to_line = np.linalg.norm(vec_to_line, axis=1)
        return np.argmax(dist_to_line)
    
    def query(self, query_text: str, n_matches: int) -> list:
        query_embedding = self.model.encode([query_text])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings).flatten()
        sorted_indices = np.argsort(similarities)[::-1]
        return [self.documents[i] for i in sorted_indices[:n_matches]]

# Usage
if __name__ == "__main__":
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    kmr = KMR(url)
    print(kmr.query("chain of thought", 5))
