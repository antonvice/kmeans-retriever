import numpy as np
import re
from bs4 import BeautifulSoup
import aiohttp
from sentence_transformers import SentenceTransformer
import asyncio
from ollama import AsyncClient
import matplotlib.pyplot as plt
from semantic_text_splitter import HuggingFaceTextSplitter
from tokenizers import Tokenizer
import heapq
from collections import deque, defaultdict
import functools

class Vice:

    def __init__(self, embedding_model_name='BAAI/bge-large-zh-v1.5', chat_model_name="gemma:2b-instruct", max_rag_length=3000):
        self.embedding_model = SentenceTransformer(embedding_model_name, device="mps:0")
        self.chat_model_name = chat_model_name
        self.messages = []
        self.client = AsyncClient()
        self.tokenizer = Tokenizer.from_pretrained("BAAI/bge-large-zh-v1.5")
        self.splitter = HuggingFaceTextSplitter(self.tokenizer, trim_chunks=True)
        self.documents = {}
        self.max_rag_length = max_rag_length
        self.cache = defaultdict(dict)
        
    @classmethod
    def with_default_models(cls):
        """
        A class method that initializes the class with default embedding and chat models.
        """
        default_embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
        default_chat_model = 'gemma:2b-instruct'
        return cls(embedding_model_name=default_embedding_model, chat_model_name=default_chat_model)
    
    @staticmethod
    def is_valid_url(url):
        """
        A static method to validate a URL using a simple regex pattern.
        
        Parameters:
            url (str): The URL to be validated.
            
        Returns:
            bool: True if the URL is valid, False otherwise.
        """
        import re
        # This is a simple regex pattern for demonstration; consider using more robust validation
        pattern = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')
        return re.match(pattern, url) is not None

    @staticmethod
    def clean_text(text):
        """
        A static method to clean the input text by removing extra spaces and newline characters.

        Parameters:
            text (str): The input text to be cleaned.

        Returns:
            str: The cleaned text with extra spaces and newline characters removed.
        """
        # Simple example: remove extra spaces and newline characters
        cleaned_text = ' '.join(text.split())
        return cleaned_text
    
    def init_chat(self, system_prompt):
        """
        Initializes the chat system with the given system prompt.

        :param system_prompt: The system prompt to initialize the chat with.
        :return: None
        """
        self.messages = [{'role': 'system', 'content': system_prompt}]
        self.client = AsyncClient()

    def print_history(self, role=None, keyword=None):
        """
        Print the chat history, including the role and content of each message.
        """
        for message in self.messages:
            if role and message['role'] != role:
                continue
            if keyword and keyword not in message['content']:
                continue
            print(f"{message['role']}: {message['content']}\n")
        

    async def get_response(self, prompt):
        """
        Sends a message to the chat system and returns the response.

        :param client: The client to use for the chat.
        :param messages: The messages to send to the chat system.
        :return: The response from the chat system.
        """
        context = self.get_similar_chunks(prompt)
        
        message = {"role": "user", "content": prompt+"\n=== CONTEXT ===\n " + str(context)}
        self.messages.append(message)
        response = await self.client.chat(model = self.chat_model_name, messages = self.messages)
        ai_message = {"role": "assistant", "content": response['message']['content']}
        self.messages.append(ai_message)
        print(response)
        return ai_message
    
    async def _url2txt(self, url):
        """
        This function takes a URL as input, retrieves the text content from the URL, and returns the concatenated text from all the paragraphs in the HTML content.
        """
        if 'url_texts' not in self.cache:
            self.cache['url_texts'] = {}
        if url in self.cache['url_texts']:
            return self.cache['url_texts'][url] 
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    paragraphs = [p.get_text() for p in soup.find_all('p')]
                    text = ' '.join(paragraphs)
                else:
                    text=f"Error: {response.status} {response.reason}"
                self.cache['url_texts'][url] = text
                return text
                
    async def ingest(self, source):
        """
        Ingests the source text and stores chunks and their embeddings in the dictionary with the source as the key.

        Parameters:
            source (str): The source text to ingest.
            type (str): The type of the source, default is "url".

        Returns:
            None
        """
        text = await self._url2txt(source)
        chunks = self.splitter.chunks(text, chunk_capacity=(200, 1000))
        embeddings = [self._embed(chunk) for chunk in chunks]
        self.documents[source] = {"chunks": chunks, "embeddings": embeddings}
        print(f"Total number of generated chunks from {source}: {len(chunks)}")


    def get_similar_chunks(self, query, k=3):
        """
        Get similar chunks to the given query.

        Args:
            query (object): The query for which similar chunks are to be found.
            k (int, optional): The number of similar chunks to return. Defaults to 3.

        Returns:
            list: A list of similar chunks to the query.
        """
        query_embedding = self._embed(query).reshape(1, -1)  # Ensure query embedding is 2D for batch operation
        
        # Aggregate all chunk embeddings and sources
        all_embeddings = []
        source_chunk_indices = []  # Keep track of source and chunk index for each embedding
        for source, data in self.documents.items():
            all_embeddings.extend(data["embeddings"])
            source_chunk_indices.extend([(source, i) for i in range(len(data["embeddings"]))])

        all_embeddings = np.vstack(all_embeddings)  # Convert list of embeddings to a 2D array

        # Calculate cosine similarities in batch
        similarities = np.dot(all_embeddings, query_embedding.T).squeeze()  # Result is 1D array of similarities
        
        # Use a heap to maintain top k similarities
        max_heap = []
        for i, similarity in enumerate(similarities):
            source, chunk_index = source_chunk_indices[i]
            if len(max_heap) < k:
                heapq.heappush(max_heap, (similarity, source, chunk_index))
            elif similarity > max_heap[0][0]:
                heapq.heappushpop(max_heap, (similarity, source, chunk_index))

        # Sort the heap to get the top k results
        top_k_results = sorted(max_heap, reverse=True)

        # Extract the top k chunks considering the max aggregate length constraint
        selected_chunks, aggregate_length = [], 0
        for similarity, source, chunk_index in top_k_results:
            chunk = self.documents[source]["chunks"][chunk_index]
            if aggregate_length + len(chunk) > self.max_rag_length and selected_chunks:
                break
            selected_chunks.append(chunk)
            aggregate_length += len(chunk)
        
        return selected_chunks
    
    def sample(self, source, n=3):
        """
        A method that prints sample chunks from a specified source. 
        Parameters:
            - source: the source of the sample chunks
            - n: the number of chunks to print (default is 3)
        """
        if source not in self.documents:
            print("No samples found, use the ingest method to add samples.")
            return
        chunks = self.documents[source]["chunks"]
        print(f"====   Sample chunks from {source}:   ====\n")
        for i in range(n):
            print(f"### Chunk {i+1}: \n{chunks[i]}\n")
            
    @functools.lru_cache(maxsize=128)
    def _embed(self, text):
        """
        Embeds a given text (string) using the SentenceTransformer model.

        Parameters:
        - text (str): The text to embed.

        Returns:
        - A 1D numpy array representing the text embedding.
        """
        # Ensure that the text is a list of strings for the model
        text_list = [text] if isinstance(text, str) else text
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(text_list, convert_to_tensor=False)
        
        # Return the first embedding as a numpy array (since we only pass one text)
        return embeddings[0]

    def clean(self):
        """
        Method to clean up the documents attribute by resetting it to an empty dictionary.
        """
        self.documents = {}

if __name__ == "__main__":
    import asyncio
    beepbop = Vice()
    url = "https://vitalik.eth.limo/general/2023/12/28/cypherpunk.html"
    asyncio.run(beepbop.ingest(url))
    beepbop.init_chat("You are a helpful assistant.")
    beepbop.sample(url, 3)
    task = beepbop.get_response("What is the most practical use of Zupass so far?")
    asyncio.run(task)
    