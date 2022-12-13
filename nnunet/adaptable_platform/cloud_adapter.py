from google.cloud import storage

class CloudAdapter:
    def __init__(self, prefix):
        self.prefix = prefix
        return

    def DownloadBucketToFolder(self, bucket_name : str, folder_path : str):
        target_path = join(self.prefix, folder_path)
        return
