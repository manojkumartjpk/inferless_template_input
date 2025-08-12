import os
import time
from datetime import datetime

class InferlessPythonModel:

    def initialize(self):
        print("start initialize", flush=True)
        self.folder_path = os.getenv("NFS_PATH")
        self.file_path = f"{self.folder_path}/test_file.txt"

    def infer(self, inputs):
        print("start infer", flush=True)

        start_time = time.time()
        while time.time() - start_time < 20:  # loop for 20 seconds
            current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(self.file_path, 'a') as file:
                file.write("hello world " + str(current_datetime) + "\n")
            time.sleep(1)  # wait 1 second

        return {"generated_text": "text"}

    def finalize(self, args):
        print("start finalize", flush=True)
