import modal
import os

app = modal.App("nanoowl-deployment")
volume = modal.Volume.from_name("nanoowl-engines", create_if_missing=True)

ENGINE_DIR = "/root/data"

# Create image with required setup
image = (modal.Image.from_dockerfile("docker/23-01/Dockerfile")
         .pip_install([
             "torch",
             "transformers",
             "tensorrt",
             "torch2trt",
             "aiohttp",
             "opencv-python-headless"
         ])
         .run_commands(
             "cd /root && git clone https://github.com/NVIDIA-AI-IOT/nanoowl",
             "cd /root/nanoowl && python3 setup.py develop",
             f"mkdir -p {ENGINE_DIR}"
         )
         .env({
             "CUDA_VISIBLE_DEVICES": "0",
             "TORCH_CUDA_ARCH_LIST": "7.0;7.5;8.0;8.6+PTX",
             "TORCH_NVCC_FLAGS": "-Xfatbin -compress-all",
             "MAX_JOBS": "4"
         }))

@app.cls(
    image=image,
    gpu="t4",
    volumes={ENGINE_DIR: volume},
    memory=32768,  # 32GB RAM
    allow_concurrent_inputs=True,
    cpu=4,
    enable_memory_snapshot=True,
    container_idle_timeout=300  # Keep containers warm for 5 minutes
)
class NanoOwl:
    @modal.enter()
    def setup(self):
        import os
        import subprocess
        
        print("Current working directory:", os.getcwd())
        print("Directory contents:", os.listdir())
        
        # Set absolute paths
        self.engine_path = f"{ENGINE_DIR}/owl_image_encoder_patch32.engine"
        
        # Build engine if it doesn't exist
        if not os.path.exists(self.engine_path):
            subprocess.run([
                "python3",
                "-m",
                "nanoowl.build_image_encoder_engine",
                self.engine_path,
                "--fp16_mode=True"  # Enable FP16 for better performance
            ], check=True)
            volume.commit()  # Persist engine
        
        # Start the tree demo with correct absolute path
        self.process = subprocess.Popen([
            "python3",
            "/root/nanoowl/examples/tree_demo/tree_demo.py",
            self.engine_path,
            "--host", "0.0.0.0",
            "--port", "7860"
        ])

    @modal.web_endpoint()
    async def serve(self):
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
            
        # Clean up CUDA cache
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Commit any pending volume changes
        try:
            volume.commit()
        except Exception as e:
            print(f"Error during cleanup: {e}")