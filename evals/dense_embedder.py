from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer

from llama_index.core.schema import BaseNode
from torch import device as torch_device, cuda

class DenseEmbedder:
    """テキストチャンクのスパース埋め込みを生成するクラス。

    Attributes:
        None
    """

    def __init__(self, *, model_name: str = "intfloat/multilingual-e5-large", **kwargs: any):
        """TextEmbedderを初期化します。"""
        super().__init__(**kwargs)
        self.model_name = model_name
        device = torch_device("cuda" if cuda.is_available() else "cpu")
        self._model = HuggingFaceEmbeddings(model_name=self.model_name, cache_folder="embeddings", model_kwargs={"device": device})

    def count_tokens(self, text: str) -> int:
        """テキストのトークン数を返します。

        Args:
            text (str): トークン数を数えるテキスト

        Returns:
            int: テキストのトークン数
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )
        return len(tokenizer.encode(text))

    def embed_query(self, query_text: str) -> list[float]:
        """
        クエリテキストを多言語トランスフォーマーモデルを使用して埋め込み、正規化された埋め込みを返します。

        Args:
            query_text (str): 埋め込む入力クエリテキスト。

        Returns:
            list[float]: 入力クエリテキストの正規化された埋め込みのリスト。
        """
        text = ""
        if self.model_name.startswith("intfloat/multilingual-e5"):
            text = f"passage: {query_text}"
        else:
            text = query_text
        return self._model.embed_query(text=text)
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        テキストを多言語トランスフォーマーモデルを使用して埋め込み、正規化された埋め込みを返します。

        Args:
            text (str): 埋め込む入力テキスト。

        Returns:
            list[float]: 入力テキストの正規化された埋め込みのリスト。
        """
        texts = texts
        if self.model_name.startswith("intfloat/multilingual-e5"):
            texts = [f"passage: {text}" for text in texts]
        return self._model.embed_documents(texts=texts)

    def embed_nodes(self, nodes: list[BaseNode]) -> list[BaseNode]:
        """ドキュメントを多言語トランスフォーマーモデルを使用して埋め込み、正規化された埋め込みを返します。

        Args:
            documents (list[Document]): 埋め込む入力ドキュメント。

        Returns:
            list[float]: 入力ドキュメントの正規化された埋め込みのリスト。
        """
        texts = [node.get_content() for node in nodes]
        embeddings = self.embed_texts(texts=texts)
        _nodes = []
        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding
            _nodes.append(node)
        return _nodes