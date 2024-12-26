from fastembed import SparseEmbedding, SparseTextEmbedding


class TextEmbedder:
    """テキストチャンクのスパース埋め込みを生成するクラス。

    Attributes:
        None
    """

    def __init__(self, *, model_name: str = "hotchpotch/japanese-splade-v2"):
        """TextEmbedderを初期化します。"""
        self._splade_model = SparseTextEmbedding(model_name)

    def embed_documents(self, chunk_texts: list[str]) -> list[SparseEmbedding]:
        """ドキュメントに対するチャンクの埋め込みを生成します。

        Args:
            chunk_texts (list[str]): 埋め込みを生成するテキストチャンクのリスト

        Returns:
            list[SparseEmbedding]: 埋め込みのリスト
        """
        embeddings = self._splade_model.embed(documents=chunk_texts, batch_size=6)
        return embeddings

        # embeddings = self._splade_model.encode(chunk_texts)

        # results = []
        # for embedding in embeddings:
        #     token_values = self._splade_model.get_token_values(embedding=embedding)
        #     values = token_values.values()
        #     indices = token_values.keys()
        #     indices = self._splade_model.tokenizer.convert_tokens_to_ids(indices)
        #     results.append(
        #         SparseEmbedding(
        #             values=numpy.array(values), indices=numpy.array(indices)
        #         )
        #     )

        # return results

    def embed_query(self, query_text: str) -> list[SparseEmbedding]:
        embeddings = self._splade_model.embed(documents=query_text)
        return embeddings
