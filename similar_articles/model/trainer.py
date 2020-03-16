from torch.utils.data import DataLoader
from similar_articles.datasource.references_dataset import ReferencesDataset
from similar_articles.model.skipgram import SkipGram
import torch
import logging


def train(
    references_dataset: ReferencesDataset,
    output_path: str,
    epochs: int,
    embedding_size: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
):
    """
    Trains model and saves final embeddings to disk
    """
    model = SkipGram(references_dataset.vocabulary_size, embedding_size)

    for epoch in range(epochs):
        logging.info(f"epoch: {epoch + 1}")
        train_epoch(model, references_dataset, batch_size, learning_rate, num_workers)
    model.save_embedding(references_dataset.idx_to_word_map, output_path)


def train_epoch(
    model: SkipGram,
    references_dataset: ReferencesDataset,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
):
    """
    trains model for one epoch
    """

    dataloader = DataLoader(
        references_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for idx, (input_labels, positive_labels, negative_labels) in enumerate(dataloader):
        input_labels = input_labels.long()
        positive_labels = positive_labels.long()
        negative_labels = negative_labels.long()

        optimizer.zero_grad()
        loss = model(input_labels, positive_labels, negative_labels).mean()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            logging.info(f"train iteration: { idx }, loss: { loss.item() }")
