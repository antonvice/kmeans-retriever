import numpy as np
import spacy # <- deprecated
from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer
import asyncio
from ollama import AsyncClient


class BeepBop:
    def __init__(self, spacy_model_name='en_core_web_sm', embedding_model_name='BAAI/bge-large-zh-v1.5', ollama=True, chat_model_name="gemma:2b-instruct"):
        self.nlp = spacy.load(spacy_model_name)
        self.embedding_model = SentenceTransformer(embedding_model_name, device="mps:0")
        self.chat_model_name = chat_model_name
    def init_chat(self, system_prompt):
        """
        Initializes the chat system with the given system prompt.

        :param system_prompt: The system prompt to initialize the chat with.
        :return: None
        """
        self.messages = [{'role': 'system', 'content': system_prompt}]
        self.client = AsyncClient()
        
    async def get_response(self, query):
        """
        Sends a message to the chat system and returns the response.

        :param client: The client to use for the chat.
        :param messages: The messages to send to the chat system.
        :return: The response from the chat system.
        """
        message = {"role": "user", "content": query}
        self.messages.append(message)
        response = await self.client.chat(model = self.chat_model_name, messages = self.messages)
        ai_message = {"role": "assistant", "content": response['message']['content']}
        self.messages.append(ai_message)
        print(ai_message['content'])
        return ai_message

    def process(self, text):
        doc = self.nlp(text)
        sents = list(doc.sents)
        vecs = np.stack([sent.vector / sent.vector_norm for sent in sents])
        return sents, vecs

    def cluster_text(self, sents, vecs, threshold):
        clusters = [[0]]
        for i in range(1, len(sents)):
            if np.dot(vecs[i], vecs[i-1]) < threshold:
                clusters.append([])
            clusters[-1].append(i)
        return clusters

    def clean_text(self, text):
        # Add your text cleaning process here
        return text

    def fetch_and_process_url(self, url):
        """Fetches text from a URL and processes it into sentences and vectors."""
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        text = ' '.join(paragraphs)
        sents, vecs = self.process(text)
        return sents, vecs
    
    def clusterize(self, sents, vecs, threshold=0.3):
        """Clusters processed sentences based on their vector similarity."""
        clusters = self.cluster_text(sents, vecs, threshold)
        final_texts = []
        clusters_lens = []

        for cluster in clusters:
            cluster_txt = self.clean_text(' '.join([sents[i].text for i in cluster]))
            cluster_len = len(cluster_txt)

            if cluster_len < 60:
                continue
            elif cluster_len > 3000:
                # Re-cluster with adjusted threshold for long clusters
                threshold_adjusted = 0.6
                sents_div, vecs_div = self.process(cluster_txt)
                reclusters = self.cluster_text(sents_div, vecs_div, threshold_adjusted)
                
                for subcluster in reclusters:
                    div_txt = self.clean_text(' '.join([sents_div[i].text for i in subcluster]))
                    div_len = len(div_txt)
                    
                    if 60 <= div_len <= 3000:
                        clusters_lens.append(div_len)
                        final_texts.append(div_txt)
            else:
                clusters_lens.append(cluster_len)
                final_texts.append(cluster_txt)
        
        return final_texts, clusters_lens

# Example usage
if __name__ == "__main__":
    processor = BeepBop()
    url = "https://vitalik.eth.limo/general/2023/12/28/cypherpunk.html"
    sents, vecs = processor.fetch_and_process_url(url)
    final_texts, clusters_lens = processor.clusterize(sents, vecs)

    print("====   Sample chunks from 'Adjacent Sentences Clustering':   ====\n")
    for i, chunk in enumerate(final_texts[:4]):  # Display the first 4 chunks
        print(f"### Chunk {i+1}: \n{chunk}\n")

    print(f"\n\nTotal number of chunks: {len(final_texts)}")
    print(f"Average length of chunks: {np.mean(clusters_lens)}")
    processor.init_chat("You are a helpful assistant.")
    asyncio.run(processor.get_response("What is the most practical use of Zupass so far?"))
