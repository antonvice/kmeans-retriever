# Document Clustering and Retrieval System Using KMeans
This project is designed to dynamically adjust the level of abstraction in a document using Language Learning Models (LLMs). By leveraging the power of LLMs, our system allows users to interactively explore documents at various levels of detail, with the additional feature of clustering similar documents together based on their content.

# Features
* Dynamic Document Clustering: Group similar documents together using KMeans clustering and Sentence Transformers for embeddings.
* Query-based Document Retrieval: Ability to query a specific topic and retrieve documents related to that query from a cluster.
* Finds elbow curve for the optimal number of clusters
  
# Run

```bash
python -m venv .venv 
source .venv/bin/activate
pip install -r requirements.txt
python kmeans-retriever.py -i <url> -k <num_of_docks> -q <your_query>
```

# Future Improvements (🚧 TODOs)
* The idea is there, but the retrieval doesn't pull the relevant info, maybe I can classify the embedding into one of the clusters without embedding the query into the original data

# Contributions
Contributions are welcome! If you have suggestions for improvements or want to contribute to the project, please feel free to submit a pull request or open an issue.
