import MeCab
import neologdn
import stopwordsiso

from llama_index.core.schema import BaseNode

from fastembed import SparseEmbedding, SparseTextEmbedding

class BM25Embedder:
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

    def count_tokens(self, text: str) -> int:
        """テキストのトークン数を取得します。

        Args:
            text (str): トークン数を取得するテキスト

        Returns:
            int: トークン数
        """
        normalized_text = neologdn.normalize(text=text)
        tokens = self._tokenize(text=normalized_text)
        return len(tokens)

    def embed_texts(self, texts: list[str]) -> list[SparseEmbedding]:
        """ドキュメントに対するチャンクの埋め込みを生成します。

        Args:
            chunk_texts (list[str]): 埋め込みを生成するテキストチャンクのリスト

        Returns:
            list[SparseEmbedding]: 埋め込みのリスト
        """
        filtered_chunks = []
        for chunk_text in texts:
            normalized_text = neologdn.normalize(text=chunk_text)
            tokens = self._tokenize(text=normalized_text)
            concat_tokens = " ".join(tokens)
            filtered_chunks.append(concat_tokens)
        return list(self._bm25_model.embed(documents=filtered_chunks, parallel=0))
    
    
    def embed_nodes(self, nodes: list[BaseNode]) -> list[SparseEmbedding]:
        """ドキュメントを多言語トランスフォーマーモデルを使用して埋め込み、正規化された埋め込みを返します。

        Args:
            documents (list[Document]): 埋め込む入力ドキュメント。

        Returns:
            list[float]: 入力ドキュメントの正規化された埋め込みのリスト。
        """
        texts = [node.get_content() for node in nodes]
        return self.embed_texts(texts=texts)

    def embed_query(self, query: str) -> SparseEmbedding:
        """クエリに対するチャンクの埋め込みを生成します。

        Args:
            query_text (str): 埋め込みを生成するテキストチャンク

        Returns:
            SparseEmbedding: 埋め込み
        """
        normalized_text = neologdn.normalize(text=query)
        tokens = self._tokenize(text=normalized_text)
        tokenized_query = " ".join(tokens)
        return list(self._bm25_model.query_embed(query=tokenized_query))[0]
