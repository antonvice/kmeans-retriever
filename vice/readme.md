# Vice: Versatile Information Clustering Engine

## Overview
**Vice (Versatile Information Clustering Engine)** is a powerful Python class designed to facilitate the construction of Rapid Automated Gathering (RAG) systems with a focus on testing various Large Language Models (LLMs) and retrieval techniques. It leverages advanced NLP tools and models to ingest, process, cluster, and retrieve text data effectively, offering a comprehensive solution for developing efficient and scalable text-based applications.

## Features
* Flexible Text Ingestion: Supports ingesting text from URLs, with plans to extend functionality to other sources.
* Advanced Text Splitting: Utilizes the HuggingFaceTextSplitter for intelligent text chunking, ensuring optimal data preparation for processing.
* Efficient Text Embedding: Incorporates Sentence Transformers for generating high-quality text embeddings, enabling accurate similarity comparisons.
* Dynamic Text Clustering: Implements HDBSCAN clustering algorithm to group text chunks based on their semantic similarity, facilitating efficient information retrieval.
* Smart Caching: Employs caching mechanisms to enhance performance by storing frequently accessed data for quick retrieval.
* Asynchronous Support: Designed with asyncio to handle network operations and other IO-bound tasks asynchronously, improving overall efficiency.
* Interactive Chat System: Integrates with the Ollama AsyncClient for seamless interaction with chat models, enriching the user experience with context-aware responses.


Quick Start
Here's how to quickly set up and use Vice:

```
import asyncio
from vice import Vice

async def main():
    # Initialize Vice with a URL source
    vice = Vice()
    url = "https://vitalik.eth.limo/general/2023/12/28/cypherpunk.html"
    await vice.ingest(url)
    
    # Sample interaction with the chat system
    vice.init_chat("You are a helpful assistant.")
    response = await vice.get_response("What is the most practical use of Zupass so far?")
    print(response)
```