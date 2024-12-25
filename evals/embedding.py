import uuid

from datasets import load_dataset
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    NamedSparseVector,
    VectorParams,
    Distance,
)
from tqdm import tqdm
from tqdm.contrib import tenumerate


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


class TextEmbedder:
    """テキストチャンクのスパース埋め込みを生成するクラス。

    Attributes:
        None
    """

    def embed_documents(self, chunk_texts: list[str]) -> list[list[float]]:
        """ドキュメントに対するチャンクの埋め込みを生成します。

        Args:
            chunk_texts (list[str]): 埋め込みを生成するテキストチャンクのリスト

        Returns:
            list[float]: 埋め込みのリスト
        """
        model_name = "intfloat/multilingual-e5-base"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        texts = [f"passage: {chunk_text}" for chunk_text in chunk_texts]
        query_embeddings = embeddings.embed_documents(texts=texts)
        return query_embeddings

    def embed_query(self, query_text: str) -> list[float]:
        """
        クエリテキストを多言語トランスフォーマーモデルを使用して埋め込み、正規化された埋め込みを返します。

        Args:
            query_text (str): 埋め込む入力クエリテキスト。

        Returns:
            list[float]: 入力クエリテキストの正規化された埋め込みのリスト。
        """
        model_name = "intfloat/multilingual-e5-base"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs={"device": "cuda"}
        )

        text = f"query: {query_text}"
        query_embedding = embeddings.embed_query(text=text)
        return query_embedding


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

    def init_collection(self) -> None:
        """Qdrantコレクションを初期化します。

        Returns:
            None
        """
        if self.client.collection_exists(collection_name=self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    def insert_chunks(
        self, chunk_texts: list[str], dense_embeddings: list[list[float]]
    ) -> None:
        """テキストチャンクと埋め込みをQdrantにインサートします。

        Args:
            chunk_texts (list[str]): テキストチャンクのリスト
            chunk_embeddings (list[float]): 埋め込みのリスト

        Returns:
            None
        """
        points = []
        for idx, vector in tenumerate(dense_embeddings):
            id = str(uuid.uuid4())
            point = PointStruct(
                id=id,
                payload={"text": chunk_texts[idx]},
                vector=vector,
            )
            self.client.upsert(collection_name=self.collection_name, points=[point])

    def search(self, query_vector: list[float], limit: int = 5):
        """スパースベクトルを使って、キーワード検索を行います。

        Args:
            query_embedding (float): 検索に使用する埋め込み
            limit (int): 返す結果の最大数

        Returns:
            list: 検索結果のリスト。
        """
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
    qdrant_manager = QdrantManager(collection_name="dense_collection")
    qdrant_manager.init_collection()

    # Step 4: チャンクをデータベースに登録
    embedder = TextEmbedder()
    dense_embeddings = embedder.embed_documents(chunk_texts=chunk_texts)
    qdrant_manager.insert_chunks(
        chunk_texts=chunk_texts, dense_embeddings=dense_embeddings
    )

    # Step 5: クエリの埋め込み
    query_text = "越智東風君"
    query_vector = embedder.embed_query(query_text=query_text)

    # Step 6: クエリで検索
    results = qdrant_manager.search(query_vector=query_vector)
    for point in results:
        print(f"id={point.id}\nscore={point.score}\ncontent={point.payload['text']}\n")


if __name__ == "__main__":
    main()
