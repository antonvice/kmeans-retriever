# Document Clustering and Retrieval System Using KMeans
This project is designed to dynamically adjust the level of abstraction in a document using Language Learning Models (LLMs). By leveraging the power of LLMs, our system allows users to interactively explore documents at various levels of detail, with the additional feature of clustering similar documents together based on their content.

# Features
* Dynamic Document Clustering: Group similar documents together using KMeans clustering and Sentence Transformers for embeddings.
* Query-based Document Retrieval: Ability to query a specific topic and retrieve documents related to that query from a cluster.
* Finds elbow curve for the optimal number of clusters
  

# Running the Script
The script accepts three command-line arguments:

* -i: The URL of the page from which to parse and cluster documents.
* -k: The number of documents to retrieve from the query's cluster.
* -q: The query string used to find relevant documents within the cluster.

To run the script, use the following command format in your terminal:

```bash
python script_name.py -i "URL" -k NUMBER_OF_DOCS -q "QUERY"
Replace script_name.py with the actual name of your Python script file.
```
### Example:
To fetch documents from "https://lilianweng.github.io/posts/2023-06-23-agent/", cluster them, and retrieve 3 documents similar to the query "what are agents?", you would run:

```bash
python script_name.py -i "https://lilianweng.github.io/posts/2023-06-23-agent/" -k 3 -q "what are agents?"
```
Output
```
Selected documents from the query's cluster: \
['The idea of CoH is to present a history of sequentially improved \
outputs  in context and train the model to take on the trend to produce \
better outputs. Algorithm Distillation (AD; Laskin et al. 2023) applies \
the same idea to cross-episode trajectories in reinforcement learning \
tasks, where an algorithm is encapsulated in a long history-conditioned \
policy. Considering that an agent interacts with the environment many times \
and in each episode the agent gets a little better, AD concatenates this learning \
history and feeds that into the model. Hence we should expect the next predicted \
action to lead to better performance than previous trials. The goal is to learn the \
process of RL instead of training a task-specific policy itself.', 'The heuristic \
function determines when the trajectory is inefficient or contains hallucination and \
should be stopped. Inefficient planning refers to trajectories that take too long \
without success. Hallucination is defined as encountering a sequence of consecutive \
identical actions that lead to the same observation in the environment.']
```
# Future Improvements (🚧 TODOs)
* The idea is there, but the retrieval doesn't pull the relevant info, maybe I can classify the embedding into one of the clusters without embedding the query into the original data

# Contributions
Contributions are welcome! If you have suggestions for improvements or want to contribute to the project, please feel free to submit a pull request or open an issue.
