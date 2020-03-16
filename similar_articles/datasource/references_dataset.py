import numpy as np
from collections import Counter
import torch
from torch.utils.data import Dataset
from similar_articles.datasource.references_datasource import ReferencesDataSource
from typing import Dict, Tuple


class ReferencesDataset(Dataset):
    """
    """

    def __init__(
        self,
        references_datasource: ReferencesDataSource,
        context_window: int,
        max_vocabulary_size: int,
        negatives_samples: int,
        power_negatives: float,
    ):
        self.data = references_datasource.transform()
        self.context_window = context_window
        self.max_vocabulary_size = max_vocabulary_size
        self.negatives_samples = negatives_samples
        self.power_negatives = power_negatives

        self.vocabulary = self._vocabulary()
        self.word_to_idx_map = self._word_to_idx_map()
        self.idx_to_word_map = {v: k for k, v in self.word_to_idx_map.items()}
        self.vocabulary_size = len(self.vocabulary)
        self.word_frequencies = self._word_frequencies()
        self.encoded_data = self._encode_data()

    def __len__(self) -> int:
        return len(self.encoded_data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        center_word = self.encoded_data[idx]
        positive_indices = list(range(idx - self.context_window, idx)) + list(
            range(idx + 1, idx + self.context_window + 1)
        )
        positive_indices = [i % len(self.encoded_data) for i in positive_indices]
        positive_words = self.encoded_data[positive_indices]
        negative_words = torch.multinomial(
            self.word_frequencies,
            self.negatives_samples * positive_words.shape[0],
            True,
        )
        return center_word, positive_words, negative_words

    def _vocabulary(self) -> Dict:
        """
        :return:
        """
        vocabulary = dict(Counter(self.data).most_common(self.max_vocabulary_size - 1))
        vocabulary["unknown"] = len(self.data) - np.sum(list(vocabulary.values()))
        return vocabulary

    def _word_to_idx_map(self) -> Dict:
        """
        :return:
        """
        return {word: i for i, word in enumerate(self.vocabulary.keys())}

    def _word_frequencies(self) -> torch.Tensor:
        """
        :return:
        """
        word_frequencies = np.array(
            [count for count in self.vocabulary.values()], dtype=np.float32
        )
        word_frequencies = word_frequencies / np.sum(word_frequencies)
        word_frequencies = word_frequencies ** (self.power_negatives)
        word_frequencies = word_frequencies / np.sum(word_frequencies)
        return torch.Tensor(word_frequencies)

    def _encode_data(self) -> torch.Tensor:
        """
        :return:
        """
        encoded_data = [
            self.word_to_idx_map.get(word, self.vocabulary_size - 1)
            for word in self.data
        ]
        return torch.Tensor(encoded_data).long()
