import modal

app = modal.App("nanoowl-deployment")

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
             "cd /root/nanoowl && python3 setup.py develop",
             "mkdir -p /root/data"
         ))

@app.cls(
    image=image,
    gpu="t4",
)
class NanoOwl:
    @modal.enter()
    def setup(self):
        import os
        import subprocess
        
        # Set absolute paths
        self.engine_path = "/root/data/owl_image_encoder_patch32.engine"
        
        # Build engine
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
        
        # Start the tree demo with modified arguments to disable camera check at startup
        self.process = subprocess.Popen([
            "python3",
            "/root/nanoowl/examples/tree_demo/tree_demo.py",
            self.engine_path,
            "--host", "0.0.0.0",
            "--port", "7860",
            "--camera", "-1"  # Disable camera at startup
        ])
        
        # Keep the server running
        import asyncio
        while True:
            if hasattr(self, 'process'):
                if self.process.poll() is not None:
                    self.setup()
            await asyncio.sleep(1)

    @modal.exit()
    def cleanup(self):
        if hasattr(self, 'process'):
            self.process.terminate()