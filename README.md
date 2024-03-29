# TableEmbeddingsWithGNNs
The purpose of “Table Matching” is the identification of “matching” pairs inside a collection of relational tables, considering two tables a “match” if they have high similarity with respect to a similarity measure. Performing this operation on collections of millions of heterogeneous tables is hard because obtaining an exact solution is a quadratic problem. A possible way to reduce its complexity is employing blocking techniques, such as LSH, to reduce the number of comparisons. Unfortunately, the existing blocking methods are not suited to work with tables, and algorithms such as “LSH for cosine similarity” can only work with embeddings.

We approach the problem of generating table embeddings that maintain properties useful to discover matching tables, proposing two frameworks that exploit different techniques to generate such embeddings. Both of them use intermediate graph representations, the first one implements a node2vec-based approach, and the second one exploits Graph Neural Networks (GNNs).

Our experiments suggest that node2vec tends to struggle when the number of tables to embed increases. On the contrary, the GNN-based framework provided promising results when it came to processing large amounts of previously unseen data, scaling up well both in terms of execution time and embedding quality.

This repository contains the code related to the GNNTE framework, which is described in detail in [this](https://morethesis.unimore.it/theses/available/etd-09232023-174306/) master's thesis.
