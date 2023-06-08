import faiss
import numpy as np
# to use run pip install faiss-gpu

def create_approx_index(embeddings, nprobe=5):
    embedding_size = embeddings.shape[-1]

    # Number of clusters used for faiss.
    # Select a value from 4*sqrt(N) to 16*sqrt(N)
    # https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    # n_clusters = int(np.power(2, np.floor(np.log2(4 * len(embeddings)))))
    n_clusters = 4  # FIXME

    # Use Inner Product (dot-product) as Index.
    # We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
    quantizer = faiss.IndexFlatIP(embedding_size)
    index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)

    # Number of clusters to explorer at search time. We will search for nearest neighbors in 3 clusters.
    index.nprobe = nprobe

    faiss.normalize_L2(np.ascontiguousarray(embeddings))

    # Then we train the index to find a suitable clustering
    index.train(embeddings)

    # Finally we add all embeddings to the index
    index.add(embeddings)

    return index

def create_exact_index(embeddings):
    embedding_size = embeddings.shape[-1]
    
    # Use Inner Product (dot-product) as Index.
    # We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
    index = faiss.IndexFlatIP(embedding_size)
    #index = faiss.IndexFlatL2(embedding_size) 

    faiss.normalize_L2(np.ascontiguousarray(embeddings))

    # Finally we add all embeddings to the index
    index.add(embeddings)

    return index

def find_n_top_A_B_exact(embeddings_A, embeddings_B, top_k_hits):
    # embeddings_A = np.random.randn(10000,300).astype(np.float32)
    # embeddings_B = np.random.randn(10000,300).astype(np.float32)

    # Indicizzazione
    #index_A = create_exact_index(embeddings_A)
    index_B = create_exact_index(embeddings_B)

    top_k_hits = 10  # Output k hits

    queries_A = embeddings_A
    
    faiss.normalize_L2(np.ascontiguousarray(queries_A))
    distances, corpus_ids = index_B.search(queries_A, k=top_k_hits)

    return distances, corpus_ids

if __name__ == "__main__":
    find_n_top_A_B_exact(None,None,None)
    print('Done')