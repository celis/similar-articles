import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
    """
    """

    def __init__(self, vocabulary_size: int, embedding_size: int):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        self.initial_range = 0.5 / self.embedding_size
        self.out_embedding = nn.Embedding(
            self.vocabulary_size, self.embedding_size, sparse=False
        )
        self.out_embedding.weight.data.uniform_(-self.initial_range, self.initial_range)

        self.in_embedding = nn.Embedding(
            self.vocabulary_size, self.embedding_size, sparse=False
        )
        self.in_embedding.weight.data.uniform_(-self.initial_range, self.initial_range)

    def forward(self, input_labels, positive_labels, negative_labels):
        input_embedding = self.in_embedding(input_labels)
        positive_embedding = self.out_embedding(positive_labels)
        negative_embedding = self.out_embedding(negative_labels)

        log_positive = torch.bmm(positive_embedding, input_embedding.unsqueeze(2)).squeeze()
        log_negative = torch.bmm(negative_embedding, -input_embedding.unsqueeze(2)).squeeze()
        log_positive = F.logsigmoid(log_positive).sum(1)
        log_negative = F.logsigmoid(log_negative).sum(1)
        loss = log_positive + log_negative

        return -loss

    def _input_embeddings(self) -> np.array:
        """
        :return:
        """
        return self.in_embedding.weight.data.numpy()

    def save_embedding(self, idx_to_word_map: dict, path: str):
        """
        Save all embeddings to file.
        As this class only record word id, so the map from id to word has to be transfered from outside.
        """
        embeddings = self._input_embeddings()

        file = open(path, "w", encoding="utf-8")
        file.write('%d %d\n' % (len(idx_to_word_map) - 1, self.embedding_size))
        for idx, word in idx_to_word_map.items():
            if word != "unknown":
                embedding = embeddings[idx]
                embedding = " ".join(map(lambda x: str(x), embedding))
                file.write("%s %s\n" % (word, embedding))
