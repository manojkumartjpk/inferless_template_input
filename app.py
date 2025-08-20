import os
import time
import uuid
from datetime import datetime

class InferlessPythonModel:

    def initialize(self):
        print("start initialize", flush=True)
        self.folder_path = os.getenv("NFS_PATH")    
        # Create the folder if it does not exist
        if self.folder_path:
            os.makedirs(self.folder_path, exist_ok=True)
    
        # Generate a unique identifier for this pod instance
        self.pod_id = str(uuid.uuid4())[:8]  # short 8-char ID
        print("id-->" + str(self.pod_id), flush=True)
        self.file_path = f"{self.folder_path}/test_file.txt"

    def infer(self, inputs):
        print("start infer", flush=True)

        for _ in range(20):  # loop exactly 20 times
            current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(self.file_path, 'a') as file:
                file.write(f"[{self.pod_id}] hello world {current_datetime}\n")
            time.sleep(0.1)

        return {"generated_text": "text"}

    def finalize(self, args):
        print("start finalize", flush=True)
