from langchain_huggingface.embeddings import HuggingFaceEmbeddings


class TextEmbedder:
    """テキストチャンクのスパース埋め込みを生成するクラス。

    Attributes:
        None
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        """TextEmbedderを初期化します。

        Args:
            None
        """
        self.model_name = model_name

    def embed_documents(self, chunk_texts: list[str]) -> list[list[float]]:
        """ドキュメントに対するチャンクの埋め込みを生成します。

        Args:
            chunk_texts (list[str]): 埋め込みを生成するテキストチャンクのリスト

        Returns:
            list[float]: 埋め込みのリスト
        """
        embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

        texts = []
        if self.model_name.startswith("intfloat/multilingual-e5"):
            texts = [f"passage: {chunk_text}" for chunk_text in chunk_texts]
        else:
            texts = [chunk_text for chunk_text in chunk_texts]
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
        embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

        text = ""
        if self.model_name.startswith("intfloat/multilingual-e5"):
            text = f"passage: {query_text}"
        else:
            text = query_text
        query_embedding = embeddings.embed_query(text=text)
        return query_embedding
