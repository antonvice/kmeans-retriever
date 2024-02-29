import numpy as np
import re
from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer
import asyncio
from ollama import AsyncClient
import hdbscan
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class BeepBop:
    def __init__(self, embedding_model_name='BAAI/bge-large-zh-v1.5', ollama=True, chat_model_name="gemma:2b-instruct"):
        self.embedding_model = SentenceTransformer(embedding_model_name, device="mps:0")
        self.chat_model_name = chat_model_name
        self.messages = []
        self.documents = []

    def print_history(self):
        #print nicely as a chat history
        for message in self.messages:
            print(f"{message['role']}: {message['content']}\n")

    def init_chat(self, system_prompt):
        """
        Initializes the chat system with the given system prompt.

        :param system_prompt: The system prompt to initialize the chat with.
        :return: None
        """
        self.messages = [{'role': 'system', 'content': system_prompt}]
        self.client = AsyncClient()
        
    def show_samples(self, n=3):
        # print n samples from the dataset
        try:
            print("====   Sample chunks from 'Adjacent Sentences Clustering':   ====\n")
            for i in range(n):
                print(f"### Chunk {i+1}: \n{self.documents[i]}\n")
        except:
            print("No samples found, use the source_docs method to add samples.")

    def source_docs(self, source, type="url"):
        if type == "url":
            sents, vecs = self._fetch_and_process_url(source)
            self._clusterize_and_split_sentences(sents, vecs)
            print(f"Total number of generated chunks: {len(self.documents)}")
        #! TODO: implement other types of sources

    async def get_response(self, query):
        """
        Sends a message to the chat system and returns the response.

        :param client: The client to use for the chat.
        :param messages: The messages to send to the chat system.
        :return: The response from the chat system.
        """
        context = self.get_similar_chunks(query)
        message = {"role": "user", "content": str(query) + "\n=== CONTEXT ===\n " + str(context)}
        self.messages.append(message)
        response = await self.client.chat(model = self.chat_model_name, messages = self.messages)
        ai_message = {"role": "assistant", "content": response['message']['content']}
        self.messages.append(ai_message)
        print(message)
        print(response)
        print(ai_message['content'])
        return ai_message

    def _process(self, text):
        sents = re.split(r'(?<=[.!?]) +', text)
        vecs = self.embedding_model.encode(sents)
        return sents, vecs

    def _segment_and_embed(self, text):
        # Split the text into sentences and generate embeddings
        sentences = re.split(r'(?<=[.!?]) +', text)
        embeddings = self.embedding_model.encode(sentences)
        return sentences, embeddings
    

    def _fetch_and_process_url(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        text = ' '.join(paragraphs)
        sentences, embeddings = self._segment_and_embed(text)
        return sentences, embeddings
    


    def _clusterize_and_split_sentences(self, sentences, embeddings):
        # Cluster embeddings using HDBSCAN
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
        clusterer.fit(embeddings)
        
        # Reset self.documents for new clustering
        self.documents = []
        
        # Iterate through unique cluster labels
        for label in set(clusterer.labels_):
            if label == -1:
                continue  # Skip noise
            
            # Get indices of sentences in the current cluster
            indices = [i for i, lbl in enumerate(clusterer.labels_) if lbl == label]
            
            # Fetch sentences for the current cluster
            cluster_sentences = [sentences[idx] for idx in indices]
            
            # Store cluster information
            self.documents.append({"cluster": label, "sentences_in_cluster": cluster_sentences})

    def _clean_text(self, text):

        return text

    def _embed(self, text):
        return self.embedding_model.encode(text)
    
    def get_similar_chunks(self, query):
        query_embedding = self._embed(query)
        
        # Compute centroid for each cluster
        centroids = [np.mean(self._embed(cluster["sentences_in_cluster"]), axis=0) for cluster in self.documents]
        
        # Compute similarities between the query and each cluster centroid
        similarities = [np.dot(query_embedding, centroid) / (np.linalg.norm(query_embedding) * np.linalg.norm(centroid)) for centroid in centroids]
        
        # Find the index of the most similar cluster
        most_similar_idx = np.argmax(similarities)
        
        # Return sentences from the most similar cluster
        return self.documents[most_similar_idx]["sentences_in_cluster"]

    def visualize_clusters(self, embeddings, sentences):
        """
        Visualizes the sentence embeddings and their corresponding cluster memberships.

        Parameters:
        - embeddings: The embeddings of the sentences.
        - sentences: The original list of sentences.
        """
        # Reduce the dimensionality for visualization
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)

        # Fit HDBSCAN again for cluster memberships (this step could be optimized by reusing the previous fit)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
        clusterer.fit(embeddings)

        # Plot
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusterer.labels_, cmap='Spectral', s=50)
        plt.title('Clusters of Sentences')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')

        # Annotate points with the first few words of each sentence for context
        for i, txt in enumerate(sentences):
            plt.annotate(txt[:10], (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8)

        # Add a legend
        plt.legend(handles=scatter.legend_elements()[0], labels=set(clusterer.labels_), title="Clusters")

        plt.show()

if __name__ == "__main__":
    processor = BeepBop(embedding_model_name='sentence-transformers/all-MiniLM-L6-v2')  # Example model
    url = "https://vitalik.eth.limo/general/2023/12/28/cypherpunk.html"

    # Process the documents from the URL
    #processor.source_docs(url)
    sentences, embeddings = processor._fetch_and_process_url(url)
    processor.visualize_clusters(embeddings, sentences)

    # Display basic information about the processed documents
    print(f"\n\nTotal number of clusters: {len(processor.documents)}")
    print(f"Average length of clusters: {np.mean([len(cluster['sentences_in_cluster']) for cluster in processor.documents])}")

    # Show a few sample clusters
    processor.show_samples(3)

    # Initialize the chat system
    processor.init_chat("You are a helpful assistant.")

    query = "What is the most practical use of Zupass so far?"
    asyncio.run(processor.get_response(query))

    most_similar_clusters = processor.get_similar_chunks(query)
    print(f"\nSentences from the {k} most similar clusters:")
    for i, cluster in enumerate(most_similar_clusters, start=1):
        print(f"\nCluster {i}:")
        for sentence in cluster['sentences_in_cluster']:
            print(f"  - {sentence}")