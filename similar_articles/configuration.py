import json
from typing import Dict


class Configuration:
    """
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.parameters = self._read_parameters()

    def _read_parameters(self) -> Dict:
        """
        :return:
        """
        parameters = json.load(open(self.config_path, "r"))
        return parameters

    @property
    def s3(self) -> Dict:
        """
        :return:
        """
        return self.parameters["s3"]


class ModelTrainingConfig:

    def __init__(self, model_config_path: str):
        self.model_config_path = model_config_path
        self.parameters = self._read_parameters()

    def _read_parameters(self) -> Dict:
        """
        :return:
        """
        parameters = json.load(open(self.model_config_path, "r"))
        return parameters

    @property
    def data(self) -> str:
        return self.parameters["references_data"]

    @property
    def dataset(self) -> Dict:
        """
        :return:
        """
        return {
            "context_window": self.parameters["context_window"],
            "max_vocabulary_size": self.parameters["max_vocabulary_size"],
            "negatives_samples": self.parameters["negatives_samples"],
            "power_negatives": self.parameters["power_negatives"]
        }

    @property
    def trainer(self) -> Dict:
        """
        :return:
        """
        return {
            "output_path": self.parameters["output_path"],
            "epochs": self.parameters["epochs"],
            "embedding_size": self.parameters["embedding_size"],
            "batch_size": self.parameters["batch_size"],
            "learning_rate": self.parameters["learning_rate"],
            "num_workers": self.parameters["num_workers"],
        }
