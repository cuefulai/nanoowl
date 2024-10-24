import modal

app = modal.App("nanoowl-deployment")
volume = modal.Volume.from_name("nanoowl-engines", create_if_missing=True)
ENGINE_DIR = "/root/engines"

# Create image with required setup
image = (modal.Image.from_dockerfile("docker/23-01/Dockerfile")
         .pip_install([
             "torch",
             "transformers",
             "tensorrt",
             "torch2trt"
         ])
         .run_commands(
             "cd /root && git clone https://github.com/cuefulai/nanoowl",
             "cd /root/nanoowl && python3 setup.py develop"
         ))

@app.cls(
    image=image,
    gpu="t4",
    volumes={ENGINE_DIR: volume}
)
class NanoOwl:
    @modal.enter()
    def setup(self):
        import os
        import subprocess
        
        # Set absolute paths
        self.engine_path = f"{ENGINE_DIR}/owl_image_encoder_patch32.engine"
        
        # Only build engine if it doesn't exist
        if not os.path.exists(self.engine_path):
            os.makedirs(ENGINE_DIR, exist_ok=True)
            subprocess.run([
                "python3",
                "-m",
                "nanoowl.build_image_encoder_engine",
                self.engine_path
            ], check=True)

    @modal.web_endpoint()
    async def serve(self):
        import os
        import subprocess
        import asyncio
        
        # Start the tree demo
        self.process = subprocess.Popen([
            "python3",
            "/root/nanoowl/examples/tree_demo/tree_demo.py",
            self.engine_path,
            "--host", "0.0.0.0",
            "--port", "7860",
            "--camera", "-1"
        ])
        
        # Keep the server running
        while True:
            if hasattr(self, 'process'):
                if self.process.poll() is not None:
                    # Just restart the process without rebuilding
                    self.process = subprocess.Popen([
                        "python3",
                        "/root/nanoowl/examples/tree_demo/tree_demo.py",
                        self.engine_path,
                        "--host", "0.0.0.0",
                        "--port", "7860",
                        "--camera", "-1"
                    ])
            await asyncio.sleep(1)

    @modal.exit()
    def cleanup(self):
        if hasattr(self, 'process'):
            self.process.terminate()