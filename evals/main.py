from typing import Mapping
import uuid
from datasets import load_dataset
from fastembed import SparseEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    SparseVector,
    SparseVectorParams,
    SparseIndexParams,
    Modifier,
    NamedSparseVector,
    VectorParams,
)


class DatasetLoader:
    """データセットを扱うクラス

    Attributes:
        work_title (str): 作品名
    """

    def __init__(self, *, work_title: str | None = None):
        """DatasetLoaderを初期化します。

        Args:
            work_title (str): 作品名
        """
        dataset = load_dataset(path="globis-university/aozorabunko-clean")
        if work_title:
            dataset = dataset.filter(lambda row: row["meta"]["作品名"] == work_title)
        self.dataset = dataset

    def load_texts(self) -> list[str]:
        """データセットからテキスト部分を読み込みます。

        Returns:
            list[str]: テキストデータのリスト
        """

        books = self.dataset.filter(
            lambda row: row["meta"]["文字遣い種別"] == "新字新仮名"
        )
        texts = [book["text"] for book in books["train"]]
        return texts


class TextChunker:
    """テキストをチャンクに分割するクラス。

    Attributes:
        chunk_size (int): チャンクサイズ
        chunk_overlap (int): 連続するチャンク間の重複部分のサイズ
    """

    def __init__(self, chunk_size: int = 128, chunk_overlap: int = 12):
        """TextChunkerをチャンクサイズと重複部分のサイズで初期化します。

        Args:
            chunk_size (int): チャンクサイズ
            chunk_overlap (int): 連続するチャンク間の重複部分のサイズ
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_texts(self, texts: list[str]) -> list[str]:
        """テキストをチャンクに分割します。

        Args:
            texts (list[str]): 分割する入力テキストのリスト

        Returns:
            list[str]: テキストチャンクのリスト
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=[
                "\n\n",
                "\n",
                " ",
                ".",
                ",",
                "\u200b",
                "\uff0c",
                "\u3001",
                "\uff0e",
                "\u3002",
                "",
            ],
        )
        documents = text_splitter.create_documents(texts=texts)
        chunk_texts = [document.page_content for document in documents]
        return chunk_texts


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
        sparse_vectors_config: Mapping[str, SparseVectorParams] = {},
        vectors_config: VectorParams | Mapping[str, VectorParams] = {},
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

    def insert_embeddings(
        self,
        texts: list[str],
        dense_embeddings: list[float] = [],
    ) -> None:
        """テキストチャンクと埋め込みをQdrantにインサートします。

        Args:
            collection_name (str): コレクションの名前
            chunk_texts (list[str]): テキストチャンクのリスト
            chunk_embeddings (list[SparseEmbedding]): 埋め込みのリスト

        Returns:
            None
        """

        points = []
        for idx, embedding in enumerate(dense_embeddings):
            id = uuid.uuid4()
            vector = embedding
            point = PointStruct(
                id=id,
                payload={"text": texts[idx]},
                vector=vector,
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name, points=points, batch_size=100
        )

    def insert_sparse_embeddings(
        self,
        texts: list[str],
        sparse_embeddings: list[SparseEmbedding] = [],
    ) -> None:
        """テキストチャンクと埋め込みをQdrantにインサートします。

        Args:
            collection_name (str): コレクションの名前
            chunk_texts (list[str]): テキストチャンクのリスト
            chunk_embeddings (list[SparseEmbedding]): 埋め込みのリスト

        Returns:
            None
        """
        for idx, embedding in enumerate(sparse_embeddings):
            id = uuid.uuid4()
            vector = NamedSparseVector(
                name="sparse",
                vector=SparseVector(
                    indices=embedding.indices.tolist(),
                    values=embedding.values.tolist(),
                ),
            )
            point = PointStruct(
                id=id,
                payload={"text": texts[idx]},
                vector=vector,
            )
            self.client.upsert(collection_name=self.collection_name, points=[point])

    def search(self, query_embedding: SparseEmbedding, limit: int = 5):
        """スパースベクトルを使って、キーワード検索を行います。

        Args:
            query_embedding (SparseEmbedding): 検索に使用する埋め込み
            limit (int): 返す結果の最大数

        Returns:
            list: 検索結果のリスト。
        """
        query_vector = NamedSparseVector(
            name="sparse",
            vector=SparseVector(
                indices=query_embedding.indices.tolist(),
                values=query_embedding.values.tolist(),
            ),
        )
        return self.client.search(
            collection_name=self.collection_name, query_vector=query_vector, limit=limit
        )


def main():
    # Step 1: データセットの読み込み
    loader = DatasetLoader(work_title="吾輩は猫である")
    texts = loader.load_texts()

    # Step 2: チャンクの作成
    chunker = TextChunker()
    chunk_texts = chunker.split_texts(texts=texts)

    # Step 3: Qdrant データベースの準備
    qdrant_manager = QdrantManager(collection_name="eval_collection")
    qdrant_manager.init_collection(
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False), modifier=Modifier.IDF
            )
        }
    )

    chunk_embeddings = embedder.embed_documents(chunk_texts=chunk_texts)
    qdrant_manager.insert_embeddings(
        texts=chunk_texts, dense_embeddings=chunk_embeddings
    )

    # Step 5: クエリの埋め込み
    query_text = "吾輩は猫である"
    query_embedding = embedder.embed_query(query_text=query_text)

    # Step 6: クエリで検索
    results = qdrant_manager.search(query_embedding=query_embedding)
    for point in results:
        print(f"id={point.id}\nscore={point.score}\ncontent={point.payload['text']}\n")


if __name__ == "__main__":
    main()
