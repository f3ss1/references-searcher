from pathlib import Path
import numpy as np

from gensim.models import Word2Vec

from references_searcher.exceptions import NotFittedError
from references_searcher.utils import verbose_iterator, log_with_message
from references_searcher.models import BasicPreprocessor


# Assuming the sentences are already processed, meaning lemmatized + filtered + lowercased, etc.
class Word2VecEmbeddings:
    def __init__(
        self,
        vector_size: int = 100,
        window_size: int = 5,
        n_workers: int = -1,
        min_count: int = 0,
        random_seed: int = 42,
    ) -> None:
        self.vector_size = vector_size
        self.window_size = window_size
        self.n_workers = n_workers
        self.min_count = min_count
        self.w2v = None
        self.random_seed = random_seed

        self.preprocessor = BasicPreprocessor()

    @log_with_message("fitting word2vec model")
    def fit(
        self,
        sentences: list[str],
    ) -> None:

        processed_sentences = self.preprocessor.transform(sentences)
        self.w2v = Word2Vec(
            sentences=processed_sentences,
            vector_size=self.vector_size,
            window=self.window_size,
            workers=self.n_workers,
            min_count=self.min_count,
            seed=self.random_seed,
            sg=1,
        )

    @log_with_message("constructing embeddings with word2vec model", log_level="DEBUG")
    def transform(
        self,
        sentences: list[str] | str,
        verbose: bool = True,
    ) -> np.ndarray:
        if self.w2v is None:
            error_message = "w2v model was not fit while being asked for a word transform!"
            raise NotFittedError(error_message)

        if isinstance(sentences, str):
            sentences = [sentences]

        processed_sentences = self.preprocessor.transform(sentences)
        pbar = verbose_iterator(processed_sentences, verbose, desc="Creating w2v embeddings", leave=False)
        resulting_embeddings = []

        for sentence in pbar:
            sentence_embedding = sum([self._transform_token(token) for token in sentence]) / len(sentence)
            resulting_embeddings.append(sentence_embedding)

        return np.array(resulting_embeddings)

    def fit_transform(
        self,
        sentences: list[str] | str,
        verbose: bool = False,
    ) -> np.ndarray:
        self.fit(sentences)
        return self.transform(sentences, verbose)

    def save_model(
        self,
        file_path: str | Path,
    ):
        if isinstance(file_path, Path):
            file_path = str(file_path)

        if self.w2v:
            self.w2v.save(file_path)
        else:
            raise NotFittedError("w2v model was not fit while being asked to save it!")

    def load_model(
        self,
        file_path: str | Path,
    ):
        if isinstance(file_path, Path):
            file_path = str(file_path)

        self.w2v = Word2Vec.load(file_path)

    def _transform_token(
        self,
        token: str,
    ) -> np.ndarray:

        if token in self.w2v.wv:
            return self.w2v.wv[token]
        else:
            return np.zeros(self.vector_size)


if __name__ == "__main__":
    embedder = Word2VecEmbeddings()
    embedder.fit(["cat walks on the floor", "man lie"])
    print(embedder.transform(["cat walk floor", "man lie"]))
