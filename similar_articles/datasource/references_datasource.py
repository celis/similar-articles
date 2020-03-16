from typing import List


class ReferencesDataSource:
    """
    Class responsible for reading references data from disk
    """

    def __init__(self, path: str):
        self.path = path

    def transform(self) -> List:
        """
        Returns list with collection of references
        """
        with open(self.path) as file:
            recids = file.readlines()
            recids = [line.strip("\n").split(",") for line in recids]
            return [recids for sublist in recids for recids in sublist]
