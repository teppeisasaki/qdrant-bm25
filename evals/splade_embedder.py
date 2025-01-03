from fastembed import SparseEmbedding
from llama_index.core.schema import BaseNode

from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

class SpladeEmbedder():
    """テキストチャンクのスパース埋め込みを生成するクラス。

    Attributes:
        None
    """

    def __init__(self, *, model_name: str = "hotchpotch/japanese-splade-v2", **kwargs: any):
        """TextEmbedderを初期化します。"""
        self.model_name = model_name
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tqdm.write(f"Using device: {device}")
        self._model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    
    def embed_texts(self, texts: list[str], batch_size: int = 1) -> list[SparseEmbedding]:
        """
        テキストを多言語トランスフォーマーモデルを使用して埋め込み、正規化された埋め込みを返します。

        Args:
            text (str): 埋め込む入力テキスト。

        Returns:
            list[float]: 入力テキストの正規化された埋め込みのリスト。
        """
        def splade_max_pooling(logits, attention_mask):
            relu_log = torch.log(1 + torch.relu(logits))
            weighted_log = relu_log * attention_mask.unsqueeze(-1)
            max_val, _ = torch.max(weighted_log, dim=1)
            return max_val
        
        # Preallocate sparse_vectors with None to maintain order
        sparse_vectors = [None] * len(texts)
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]

            tokens = self._tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            tokens = {k: v.to(self._model.device) for k, v in tokens.items()}

            with torch.no_grad():
                outputs = self._model(**tokens)

            embeddings = splade_max_pooling(outputs.logits, tokens["attention_mask"])

            for j, embedding in enumerate(embeddings):
                indices = torch.nonzero(embedding, as_tuple=True)[0]
                values = embedding[indices]
                sparse_vectors[start_idx + j] = SparseEmbedding(indices=indices, values=values)

        return sparse_vectors

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
        """クエリを多言語トランスフォーマーモデルを使用して埋め込み、正規化された埋め込みを返します。

        Args:
            query (str): 埋め込む入力クエリ。

        Returns:
            list[float]: 入力クエリの正規化された埋め込みのリスト。
        """
        return self.embed_texts(texts=[query])[0]