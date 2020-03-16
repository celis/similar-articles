import boto3
from similar_articles.configuration import Configuration


class S3:
    def __init__(self, config: Configuration):
        self.config = config

    def _boto3_client(self):
        """
        :return:
        """
        boto3_client = boto3.client(
            "s3",
            region_name=self.config.s3["region_name"],
            aws_access_key_id=self.config.s3["access_key"],
            aws_secret_access_key=self.config.s3["secret_key"],
        )
        return boto3_client

    def download(self, key: str, filename: str):
        """
        key: The name of the key to download from.
        filename: The path to the file to download to.
        """
        boto3_client = self._boto3_client()
        boto3_client.download_file(self.config.s3["bucket"], key, filename)

    def upload(self, filename: str, key: str):
        """
        filename: The path to the file to upload.
        key: The name of the key to upload to.
        """
        boto3_client = self._boto3_client()
        boto3_client.upload_file(filename, self.config.s3["bucket"], key)
