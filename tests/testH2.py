# tests/test.py
import sys
import os

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from kmeans_retriever import *

## Main


def main(url: str, query: str, num_docs: int):
    # Load the Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Fetch HTML content and parse it into paragraphs
    html_content = fetch_html_content(url)
    documents = parse_html_to_paragraphs(html_content)
    
    # Testing hyp 2
    
    # Encode all documents without the query
    embeddings = encode_documents(model, documents)
    
    # Perform clustering on the embeddings
    kmeans = cluster_documents(embeddings)
    
    # Query embedding
    query_embedding = model.encode(query)
    
    # Find the query's cluster
    query_index = kmeans.predict(query_embedding.reshape(1, -1))[0]
    
    # Retrieve indices of documents from the query's cluster
    selected_indices = retrieve_docs_from_query_cluster(kmeans, query_index, num_docs)
    
    # Print the selected documents
    for index in selected_indices:
        print(documents[index])

if __name__ == "__main__":
    import argparse
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
