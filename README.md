# Document Clustering and Retrieval System Using KMeans
This project is designed to test out a KMEANS clustering of textual embeddings of a document in order to split the text and reduce token usage and forgetting of RAG systems

# Field notes:
## V 0.1 - add query to the documents, calculate elbow curve, build clusters, retrieve k docs from the query cluster.
First iteration of the semantic clustering turned out ineffective, moreover, the retrieved documents would be different on each run. I believe that there can be two possible explainations
1. h0 no change, H1 - I am returning unsorted documents, I need to query K cosine similar documents from the cluster of query
- TestH1 shows promises, the wording is similar to the query, now let's see how it performs compared to all docs

2. H0 no change, H2 - embedding query with the clusters interferes with the latent space. I need to first cluster, then classify the query
- TestH2 confirmed H2 hypotheses, clustering without the query returns different results

3. H0 no change, H3 - similar documents retrieved from clusters should be the same as if without clusters
- H3 is confirmed with retrieved documents being Exactly the same

## Conclusion of tests
Tests verified that the technique I came up with for clustering the documents based on KMEANS clustering is an efficient text splitting technique for splitting large amounts of texts from a document for future text retrieval for Retrieval Augmented Generation SYstems, I will polish the algorithm and utilize it in my future work

# Features
* Dynamic Document Clustering: Group similar documents together using KMeans clustering and Sentence Transformers for embeddings.
* Query-based Document Retrieval: Ability to query a specific topic and retrieve documents related to that query from a cluster.
* Finds elbow curve for the optimal number of clusters

# Running the Script
The script accepts three command-line arguments:

* -i: The URL of the page from which to parse and cluster documents.
* -k: The number of documents to retrieve from the query's cluster.
* -q: The query string used to find relevant documents within the cluster.

### OR 

* -t: opt out for automatic testing with precoded arguments

### Example:
To fetch documents from "https://lilianweng.github.io/posts/2023-06-23-agent/", cluster them, and retrieve 3 documents similar to the query "what are agents?", you would run:

```bash
python kmeans_retriever.py -i "https://lilianweng.github.io/posts/2023-06-23-agent/" -k 3 -q "Chain of Thought?"
#or
python kmeans_retriever.py -t
```
Output
```
['Chain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.', 'Tree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.', '[1] Wei et al. “Chain of thought prompting elicits reasoning in large language models.” NeurIPS 2022']
```
# Future Improvements (🚧 TODOs)
* The idea is there, but the retrieval doesn't pull the relevant info, maybe I can classify the embedding into one of the clusters without embedding the query into the original data

# Contributions
Contributions are welcome! If you have suggestions for improvements or want to contribute to the project, please feel free to submit a pull request or open an issue.
