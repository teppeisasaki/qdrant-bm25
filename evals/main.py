import uuid
from datasets import load_dataset

from langchain_text_splitters import RecursiveCharacterTextSplitter

from llama_index.core.node_parser import LangchainNodeParser
from llama_index.core.schema import MediaResource, Document

from qdrant_client import QdrantClient
from qdrant_client.models import SparseVectorParams, VectorParams, Distance, PointStruct, SparseVector, NamedSparseVector

from dense_embedder import DenseEmbedder
from splade_embedder import SpladeEmbedder
from bm25_embedder import BM25Embedder

from tqdm import tqdm
from tqdm.contrib import tzip

class QdrantManager:
    """Qdrantコレクションと操作を管理するクラス。

    Attributes:
        client (QdrantClient): Qdrantクライアント
    """

    def __init__(self, *, url: str = "http://localhost:6333", collection_name: str):
        """Qdrantクライアントとコレクション名でQdrantManagerを初期化します。

        Args:
            url (str): QdrantデータベースのURL
        """
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name

    def init_collection(
        self,
        sparse_vectors_config: dict[str, SparseVectorParams] = {},
        vectors_config: VectorParams | dict[str, VectorParams] = {},
    ) -> None:
        """Qdrantコレクションを初期化します。

        Returns:
            None
        """
        if self.client.collection_exists(collection_name=self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )

    def insert_points(
        self,
        points: list[PointStruct] = [],
        batch_size: int = 1,
    ) -> None:
        """テキストチャンクと埋め込みをQdrantにインサートします。

        Args:
            collection_name (str): コレクションの名前
            chunk_texts (list[str]): テキストチャンクのリスト
            chunk_embeddings (list[SparseEmbedding]): 埋め込みのリスト

        Returns:
            None
        """
        for i in tqdm(range(0, len(points), batch_size), desc="Inserting points"):
            self.client.upsert(
                collection_name=self.collection_name, points=points[i:i + batch_size])

    def search(self, query_vector: list[float] | NamedSparseVector, limit: int = 5):
        """スパースベクトルを使って、キーワード検索を行います。

        Args:
            query_embedding (SparseEmbedding): 検索に使用する埋め込み
            limit (int): 返す結果の最大数

        Returns:
            list: 検索結果のリスト。
        """
        return self.client.search(
            collection_name=self.collection_name, query_vector=query_vector, limit=limit
        )
    
    def close(self):
        self.client.close()


def main(k: int = 5):
    # Loading dataset
    dataset = load_dataset(path="kumapo/JAQKET", name="v1.0", split="validation", cache_dir="ds_cache", trust_remote_code=True)
    # dataset = load_dataset(path="shunk031/JGLUE", name="JSQuAD", split="validation", cache_dir="ds_cache", trust_remote_code=True)
    dataset = dataset.select(range(2**7))
    qids = dataset["id"]
    questions = dataset["question"]
    contexts = dataset["context"]



    # Creating documents
    # Questions
    question_documents = []
    for qid, question_document in tzip(qids, questions, desc="Creating documents for questions"):
        id = str(uuid.uuid4())
        question_documents.append(Document(id_=id, text_resource=MediaResource(text=question_document), metadata={"qid": qid}))
    # Contexts
    context_documents = []
    for qid, context in tzip(qids, contexts, desc="Creating documents for contexts"):
        for text in context:
            id = str(uuid.uuid4())
            context_documents.append(Document(id_=id, text_resource=MediaResource(text=text), metadata={"qid": qid}))

    # Chunking texts
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=52,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ]
    )
    parser = LangchainNodeParser(text_splitter)
    chunked_context_nodes = parser.get_nodes_from_documents(documents=context_documents, show_progress=True)

    # MARK: - Embedding
    # Dense Embedding
    dense_embedder = DenseEmbedder(model_name="intfloat/multilingual-e5-small")
    dense_nodes = dense_embedder.embed_nodes(chunked_context_nodes)

    # Create Qdrant collection
    dense_collection = QdrantManager(collection_name="dense")
    VECTORS_CONFIG = VectorParams(size=384, distance=Distance.COSINE)
    dense_collection.init_collection(vectors_config=VECTORS_CONFIG)

    # Insert embeddings
    points = []
    for embedding in tqdm(dense_nodes):
        point = PointStruct(
            id=str(uuid.uuid4()),
            payload={"text": embedding.get_content(), "qid": embedding.metadata["qid"]},
            vector=embedding.embedding,
        )
        points.append(point)
    dense_collection.insert_points(points=points, batch_size=100)

    # Evaluate MRR@k
    mrr = 0
    for question_document in tqdm(question_documents, desc=f"Evaluating MRR@{k}"):
        query = question_document.text_resource.text
        query_vector = dense_embedder.embed_query(query)
        retrieved_chunks = dense_collection.search(
            query_vector=query_vector, limit=k
        )
        for i, chunk in enumerate(retrieved_chunks):
            if chunk.payload["qid"] == question_document.metadata["qid"]:
                mrr += 1 / (i + 1)
                break
    mrr /= len(question_documents)

    # Evaluate Hit Rate@k
    hit_rate = 0
    for question_document in tqdm(question_documents, desc=f"Evaluating Hit Rate@{k}"):
        query = question_document.text_resource.text
        query_vector = dense_embedder.embed_query(query)
        retrieved_chunks = dense_collection.search(
            query_vector=query_vector, limit=k
        )
        for chunk in retrieved_chunks:
            if chunk.payload["qid"] == question_document.metadata["qid"]:
                hit_rate += 1
                break
    hit_rate /= len(question_documents)

    # Store results
    results = {}
    results["Dense Embedding"] = {"MRR@k": mrr, "Hit Rate@k": hit_rate}
    
    # Close collection
    dense_collection.close()


    # MARK: - Sparse Embedding
    sparse_embedder = SpladeEmbedder(model_name="hotchpotch/japanese-splade-v2")
    sparse_embeddings = sparse_embedder.embed_nodes(chunked_context_nodes)

    # Create Qdrant collection
    sparse_collection = QdrantManager(collection_name="sparse")
    SPARSE_VECTORS_CONFIG = {
        "sparse": SparseVectorParams()
    }
    sparse_collection.init_collection(sparse_vectors_config=SPARSE_VECTORS_CONFIG)

    # Insert embeddings
    points = []
    for node, embedding in tzip(chunked_context_nodes, sparse_embeddings):
        point = PointStruct(
            id=str(uuid.uuid4()),
            payload={"text": node.get_content(), "qid": node.metadata["qid"]},
            vector={
                "sparse": SparseVector(
                    indices=embedding.indices.tolist(),
                    values=embedding.values.tolist()
                )
            },
        )
        points.append(point)
    sparse_collection.insert_points(points=points, batch_size=100)

    # Evaluate MRR@k
    mrr = 0
    for question_document in tqdm(question_documents, desc=f"Evaluating MRR@{k}"):
        query = question_document.text_resource.text
        query_embedding = sparse_embedder.embed_query(query)
        query_vector = NamedSparseVector(
            name="sparse",
            vector=SparseVector(
                indices=query_embedding.indices.tolist(),
                values=query_embedding.values.tolist()
            )
        )
        retrieved_chunks = sparse_collection.search(
            query_vector=query_vector,
            limit=k
        )
        for i, chunk in enumerate(retrieved_chunks):
            if chunk.payload["qid"] == question_document.metadata["qid"]:
                mrr += 1 / (i + 1)
                break
    mrr /= len(question_documents)

    # Evaluate Hit Rate@k
    hit_rate = 0
    for question_document in tqdm(question_documents, desc=f"Evaluating Hit Rate@{k}"):
        query = question_document.text_resource.text
        query_embedding = sparse_embedder.embed_query(query)
        query_vector = NamedSparseVector(
            name="sparse",
            vector=SparseVector(
                indices=query_embedding.indices.tolist(),
                values=query_embedding.values.tolist()
            )
        )
        retrieved_chunks = sparse_collection.search(
            query_vector=query_vector,
            limit=k
        )
        for chunk in retrieved_chunks:
            if chunk.payload["qid"] == question_document.metadata["qid"]:
                hit_rate += 1
                break
    hit_rate /= len(question_documents)

    # Store results
    results["Sparse Embedding"] = {"MRR@k": mrr, "Hit Rate@k": hit_rate}

    # Close collection
    sparse_collection.close()


    # MARK: - BM25 Embedding
    bm25_embedder = BM25Embedder()
    bm25_embeddings = bm25_embedder.embed_nodes(chunked_context_nodes)

    # Create Qdrant collection
    bm25_collection = QdrantManager(collection_name="bm25")
    SPARSE_VECTORS_CONFIG = {
        "sparse": SparseVectorParams()
    }
    bm25_collection.init_collection(sparse_vectors_config=SPARSE_VECTORS_CONFIG)

    # Insert embeddings
    points = []
    for node, embedding in tzip(chunked_context_nodes, bm25_embeddings):
        point = PointStruct(
            id=str(uuid.uuid4()),
            payload={"text": node.get_content(), "qid": node.metadata["qid"]},
            vector={
                "sparse": SparseVector(
                    indices=embedding.indices.tolist(),
                    values=embedding.values.tolist()
                )
            },
        )
        points.append(point)
    bm25_collection.insert_points(points=points, batch_size=100)

    # Evaluate MRR@k
    mrr = 0
    for question_document in tqdm(question_documents, desc=f"Evaluating MRR@{k}"):
        query = question_document.text_resource.text
        query_embedding = bm25_embedder.embed_query(query)
        query_vector = NamedSparseVector(
            name="sparse",
            vector=SparseVector(
                indices=query_embedding.indices.tolist(),
                values=query_embedding.values.tolist()
            )
        )
        retrieved_chunks = bm25_collection.search(
            query_vector=query_vector,
            limit=k
        )
        for i, chunk in enumerate(retrieved_chunks):
            if chunk.payload["qid"] == question_document.metadata["qid"]:
                mrr += 1 / (i + 1)
                break
    mrr /= len(question_documents)

    # Evaluate Hit Rate@k
    hit_rate = 0
    for question_document in tqdm(question_documents, desc=f"Evaluating Hit Rate@{k}"):
        query = question_document.text_resource.text
        query_embedding = bm25_embedder.embed_query(query)
        query_vector = NamedSparseVector(
            name="sparse",
            vector=SparseVector(
                indices=query_embedding.indices.tolist(),
                values=query_embedding.values.tolist()
            )
        )
        retrieved_chunks = bm25_collection.search(
            query_vector=query_vector,
            limit=k
        )
        for chunk in retrieved_chunks:
            if chunk.payload["qid"] == question_document.metadata["qid"]:
                hit_rate += 1
                break
    hit_rate /= len(question_documents)

    # Store results
    results["BM25 Embedding"] = {"MRR@k": mrr, "Hit Rate@k": hit_rate}

    # Print summary results
    print("\nSummary of Results:")
    for embedding_type, metrics in results.items():
        print(f"{embedding_type}:")
        print(f"  MRR@{k}: {metrics['MRR@k']}")
        print(f"  Hit Rate@{k}: {metrics['Hit Rate@k']}")

    # Close collection
    bm25_collection.close()



if __name__ == "__main__":
    import sys
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(k=k)
