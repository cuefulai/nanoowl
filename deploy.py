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
             "opencv-python-headless",
             "matplotlib",
             "pillow",
             "numpy"
         ])
         .run_commands(
             "cd /root && git clone https://github.com/NVIDIA-AI-IOT/nanoowl",
             "cd /root/nanoowl && python3 setup.py develop",
             f"mkdir -p {ENGINE_DIR}"
         ))

@app.cls(
    image=image,
    gpu="t4",
    volumes={ENGINE_DIR: volume},
    memory=32768,  # 32GB RAM
    allow_concurrent_inputs=True,
    cpu=4,
    container_idle_timeout=300  # Keep containers warm for 5 minutes
)
class NanoOwl:
    @modal.enter()
    def setup(self):
        import os
        import subprocess
        from nanoowl.owl_predictor import OwlPredictor
        from nanoowl.tree_predictor import TreePredictor
        
        # Set absolute paths
        self.engine_path = f"{ENGINE_DIR}/owl_image_encoder_patch32.engine"
        
        # Build engine if it doesn't exist
        if not os.path.exists(self.engine_path):
            subprocess.run([
                "python3", 
                "-m",
                "nanoowl.build_image_encoder_engine",
                self.engine_path,
                "--fp16_mode=True"
            ], check=True)
            volume.commit()
            
        # Initialize predictor
        self.predictor = TreePredictor(
            owl_predictor=OwlPredictor(
                image_encoder_engine=self.engine_path
            )
        )

    @modal.method()
    def process_frame(self, frame_data: str, prompt: str = None):
        import base64
        import cv2
        import numpy as np
        from PIL import Image
        from nanoowl.tree import Tree
        from nanoowl.tree_drawing import draw_tree_output
        
        # Decode base64 image
        img_data = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to PIL Image
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if prompt:
            try:
                # Process image with NanoOWL
                tree = Tree.from_prompt(prompt)
                clip_encodings = self.predictor.encode_clip_text(tree)
                owl_encodings = self.predictor.encode_owl_text(tree)
                
                detections = self.predictor.predict(
                    image_pil,
                    tree=tree,
                    clip_text_encodings=clip_encodings,
                    owl_text_encodings=owl_encodings
                )
                
                # Draw detections
                image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                image = draw_tree_output(image, detections, tree)
            except Exception as e:
                print(f"Error processing frame: {e}")
                return None
            
        # Encode processed image
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 50])
        processed_frame = base64.b64encode(buffer).decode('utf-8')
        
        return processed_frame

    @modal.web_endpoint()
    async def serve(self):
        import aiohttp
        from aiohttp import web
        import asyncio
        import logging
        import weakref
        
        app = web.Application()
        app['websockets'] = weakref.WeakSet()
        
        async def handle_index(request):
            return web.FileResponse("/root/nanoowl/examples/tree_demo/index.html")
        
        async def websocket_handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            request.app['websockets'].add(ws)
            
            prompt_data = None
            
            try:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        if "prompt:" in msg.data:
                            prompt = msg.data.split("prompt:")[1]
                            prompt_data = prompt
                        elif "frame:" in msg.data:
                            frame_data = msg.data.split("frame:")[1]
                            # Process frame and send back result
                            processed_frame = await asyncio.get_event_loop().run_in_executor(
                                None, self.process_frame, frame_data, prompt_data
                            )
                            if processed_frame:
                                await ws.send_str(f"processed_frame:{processed_frame}")
            finally:
                request.app['websockets'].discard(ws)
            return ws

        app.router.add_get("/", handle_index)
        app.router.add_get("/ws", websocket_handler)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', 7860)
        await site.start()
        
        # Keep the server running
        while True:
            await asyncio.sleep(3600)  # Sleep for an hour