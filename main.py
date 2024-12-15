import MeCab
import neologdn
import stopwordsiso
from datasets import load_dataset
from fastembed import SparseEmbedding, SparseTextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    SparseVector,
    SparseVectorParams,
    SparseIndexParams,
    Modifier,
    NamedSparseVector,
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


class TextEmbedder:
    """テキストチャンクのスパース埋め込みを生成するクラス。

    Attributes:
        None
    """

    def __init__(self):
        """TextEmbedderを初期化します。"""
        self._bm25_model = SparseTextEmbedding(
            model_name="Qdrant/bm25", disable_stemmer=True
        )
        self._mecab_tagger = MeCab.Tagger()
        self._stopwords = stopwordsiso.stopwords("ja")

    def _remove_symbols(self, nodes: list) -> list:
        """補助記号を削除します。

        Args:
            nodes (list): トークン化されたノードのリスト

        Returns:
            list: 記号を含まないノードのリスト
        """
        # ref. https://hayashibe.jp/tr/mecab/dictionary/unidic/pos
        return [node for node in nodes if node[1] != "補助記号"]

    def _remove_stopwords(self, nodes: list) -> list:
        """ストップワードを削除します。

        Args:
            nodes (list): トークン化されたノードのリスト

        Returns:
            list: ストップワードを含まないノードのリスト
        """
        return [node for node in nodes if node[0] not in self._stopwords]

    def _tokenize(self, text: str) -> list[str]:
        """MeCabを使用してテキストをトークン化します。

        Args:
            text (str): トークン化する入力テキスト

        Returns:
            list[str]: トークンのリスト
        """
        # 形態素解析
        lines = self._mecab_tagger.parse(text).splitlines()[:-1]
        nodes = [
            [line.split("\t")[0], line.split("\t")[4].split("-")[0]] for line in lines
        ]
        # 補助記号を削除
        nodes = self._remove_symbols(nodes)
        # ストップワードを削除
        nodes = self._remove_stopwords(nodes)
        return [node[0] for node in nodes]

    def embed_documents(self, chunk_texts: list[str]) -> list[SparseEmbedding]:
        """ドキュメントに対するチャンクの埋め込みを生成します。

        Args:
            chunk_texts (list[str]): 埋め込みを生成するテキストチャンクのリスト

        Returns:
            list[SparseEmbedding]: 埋め込みのリスト
        """
        filtered_chunks = []
        for chunk_text in chunk_texts:
            normalized_text = neologdn.normalize(text=chunk_text)
            tokens = self._tokenize(text=normalized_text)
            concat_tokens = " ".join(tokens)
            filtered_chunks.append(concat_tokens)
        return list(self._bm25_model.embed(documents=filtered_chunks, parallel=0))

    def embed_query(self, query_text: str) -> SparseEmbedding:
        """クエリに対するチャンクの埋め込みを生成します。

        Args:
            query_text (str): 埋め込みを生成するテキストチャンク

        Returns:
            SparseEmbedding: 埋め込み
        """
        normalized_text = neologdn.normalize(text=query_text)
        tokens = self._tokenize(text=normalized_text)
        tokenized_query = " ".join(tokens)
        return list(self._bm25_model.query_embed(query=tokenized_query))[0]


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

        sparse_config = {
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False), modifier=Modifier.IDF
            )
        }
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={},
            sparse_vectors_config=sparse_config,
        )

    def insert_chunks(
        self, chunk_texts: list[str], chunk_embeddings: list[SparseEmbedding]
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
        for idx, embedding in enumerate(chunk_embeddings):
            point = PointStruct(
                id=idx + 1,
                payload={"text": chunk_texts[idx]},
                vector={
                    "sparse": SparseVector(
                        indices=embedding.indices.tolist(),
                        values=embedding.values.tolist(),
                    )
                },
            )
            points.append(point)
        self.client.upsert(collection_name=self.collection_name, points=points)

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
    qdrant_manager = QdrantManager(collection_name="test_collection")
    qdrant_manager.init_collection()

    # Step 4: チャンクをデータベースに登録
    embedder = TextEmbedder()
    chunk_embeddings = embedder.embed_documents(chunk_texts=chunk_texts)
    qdrant_manager.insert_chunks(
        chunk_texts=chunk_texts, chunk_embeddings=chunk_embeddings
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
