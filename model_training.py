from similar_articles.datasource.references_dataset import ReferencesDataset
from similar_articles.datasource.references_datasource import ReferencesDataSource
from similar_articles.model.trainer import train
from similar_articles.configuration import ModelTrainingConfig
import logging


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    model_training_config = ModelTrainingConfig("configs/model_config.json")

    references_datasource = ReferencesDataSource(model_training_config.data)
    references_dataset = ReferencesDataset(references_datasource, **model_training_config.dataset)

    train(references_dataset, **model_training_config.trainer)



