import json
import os

import numpy as np
import torch
from transformers import pipeline
from datetime import datetime


class InferlessPythonModel:

    def initialize(self):
        print("start initialize", flush=True)
        folder_path = os.getenv("NFS_PATH")
    
        # Get current datetime and format it as YYYYMMDD_HHMMSS
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    
        # Construct the file name with the timestamp
        file_path = f"{folder_path}/test_file.txt"
    
        with open(file_path, 'a') as file:
            file.write("hello world\n")  # Write "hello world" to the end of the file
            
    def infer(self, inputs):
        print("start infer", flush=True)

        return {"generated_text": "text" }

    # perform any cleanup activity here
    def finalize(self,args):
        print("start finalize", flush=True)
