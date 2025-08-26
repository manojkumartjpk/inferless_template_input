import os
import time
import uuid

class InferlessPythonModel:

    def initialize(self):
        print("start initialize", flush=True)
        self.folder_path = os.getenv("NFS_PATH", "/tmp")  # fallback to /tmp if not set
        self.file_path = os.path.join(self.folder_path, "speed_test_file.bin")

        # Unique pod identifier
        self.pod_id = str(uuid.uuid4())[:8]
        print(f"pod_id -> {self.pod_id}", flush=True)

    def infer(self, inputs):
        print("start infer", flush=True)

        file_size_mb = 50  # test file size in MB
        data = os.urandom(1024 * 1024)  # 1MB random chunk

        # --- WRITE TEST ---
        start_time = time.time()
        with open(self.file_path, "wb") as f:
            for _ in range(file_size_mb):
                f.write(data)
        f.flush()
        os.fsync(f.fileno())
        write_time = time.time() - start_time
        write_speed = file_size_mb / write_time

        # --- READ TEST ---
        start_time = time.time()
        with open(self.file_path, "rb") as f:
            while f.read(1024 * 1024):
                pass
        read_time = time.time() - start_time
        read_speed = file_size_mb / read_time

        # Clean up
        os.remove(self.file_path)

        result = {
            "pod_id": self.pod_id,
            "file_size_MB": file_size_mb,
            "write_speed_MBps": round(write_speed, 2),
            "read_speed_MBps": round(read_speed, 2),
        }


        print(result, flush=True)
        return {"generated_text": "text"}


    def finalize(self, args):
        print("start finalize", flush=True)
