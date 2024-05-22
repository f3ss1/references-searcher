import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tag import PerceptronTagger


class BasicPreprocessor:
    ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9 ]")

    def __init__(
        self,
    ):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(nltk_stopwords.words("english"))
        self.pos_tagger = PerceptronTagger()

    def fit(self, **kwargs):
        pass

    def transform(
        self,
        sentences: str | list[str],
    ) -> list[str] | list[list[str]]:
        if isinstance(sentences, str):
            sentences = [sentences]
        result = [self._transform_sentence(sentence) for sentence in sentences]

        if len(result) == 1:
            return result[0]
        else:
            return result

    def _transform_sentence(
        self,
        sentence: str,
    ) -> list[str]:
        sentence = BasicPreprocessor.ALPHANUMERIC_PATTERN.sub(
            "",
            sentence,
        )

        tokens = word_tokenize(sentence)

        tokens, pos_tags = zip(*self.pos_tagger.tag(tokens), strict=True)
        pos_tags = [self.treebank_pos_to_wordnet_pos(pos_tag) for pos_tag in pos_tags]

        processed_tokens = self._normalize_and_remove_stopwords(
            tokens=tokens,
            pos_tags=pos_tags,
        )

        return processed_tokens

    def _normalize_and_remove_stopwords(
        self,
        tokens: list[str],
        pos_tags: list = None,
    ) -> list[str]:

        if pos_tags is not None:
            processed_tokens = [
                self.lemmatizer.lemmatize(token, pos=pos_tag)
                for token, pos_tag in zip(tokens, pos_tags, strict=False)
                if token not in self.stopwords
            ]

        else:
            processed_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stopwords]

        return processed_tokens

    @staticmethod
    def treebank_pos_to_wordnet_pos(
        treebank_pos: str,
    ) -> str:
        """Map a Treebank POS tag to a WordNet POS tag.

        Parameters
        ----------
        treebank_pos: str
            Treebank pos to map

        Returns
        -------
        A corresponding wordnet POS tag to use with WordNet-based tools.

        """
        if treebank_pos.startswith("J"):
            return wordnet.ADJ
        elif treebank_pos.startswith("V"):
            return wordnet.VERB
        elif treebank_pos.startswith("N"):
            return wordnet.NOUN
        elif treebank_pos.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN
