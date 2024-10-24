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
             "cd /root && git clone https://github.com/cuefulai/nanoowl",
             "cd /root/nanoowl && python3 setup.py develop",
             f"mkdir -p {ENGINE_DIR}"
         ))

@app.cls(
    image=image,
    gpu="t4",
    volumes={ENGINE_DIR: volume},
    memory=32768,
    allow_concurrent_inputs=True,
    cpu=4,
    container_idle_timeout=300
)
class NanoOwl:
    @modal.enter()
    def setup(self):
        import os
        import subprocess
        from nanoowl.owl_predictor import OwlPredictor
        from nanoowl.tree_predictor import TreePredictor
        
        print("Starting initialization...")
        
        self.engine_path = f"{ENGINE_DIR}/owl_image_encoder_patch32.engine"
        self.is_ready = False
        
        if not os.path.exists(self.engine_path):
            print("Building TensorRT engine (this may take a few minutes)...")
            subprocess.run([
                "python3", 
                "-m",
                "nanoowl.build_image_encoder_engine",
                self.engine_path,
                "--fp16_mode=True"
            ], check=True)
            volume.commit()
            
        print("Loading model...")
        self.predictor = TreePredictor(
            owl_predictor=OwlPredictor(
                image_encoder_engine=self.engine_path
            )
        )
        self.is_ready = True
        print("Initialization complete!")

    def get_loading_html(self):
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background: #f5f5f5;
                }
                .loading-container {
                    text-align: center;
                    padding: 20px;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .loading-spinner {
                    border: 4px solid #f3f3f3;
                    border-top: 4px solid #3498db;
                    border-radius: 50%;
                    width: 40px;
                    height: 40px;
                    animation: spin 1s linear infinite;
                    margin: 20px auto;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
            <script>
                setTimeout(() => location.reload(), 5000);
            </script>
        </head>
        <body>
            <div class="loading-container">
                <h1>Loading NanoOWL</h1>
                <div class="loading-spinner"></div>
                <p>The model is initializing. This page will refresh automatically...</p>
            </div>
        </body>
        </html>
        """

    @modal.method()
    def process_frame(self, frame_data: str, prompt: str = None):
        import base64
        import cv2
        import numpy as np
        from PIL import Image
        from nanoowl.tree import Tree
        from nanoowl.tree_drawing import draw_tree_output
        
        try:
            # Decode base64 image
            img_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert to PIL Image
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if prompt:
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
            
            # Encode processed image
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 50])
            processed_frame = base64.b64encode(buffer).decode('utf-8')
            
            return processed_frame
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None

    @modal.web_endpoint()
    async def serve(self):
        import aiohttp
        from aiohttp import web
        import asyncio
        import logging
        import weakref
        
        async def handle_index(request):
            if not hasattr(self, 'is_ready') or not self.is_ready:
                return web.Response(
                    text=self.get_loading_html(),
                    content_type='text/html'
                )
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
                            await ws.send_str("status:Prompt updated")
                        elif "frame:" in msg.data:
                            frame_data = msg.data.split("frame:")[1]
                            # Process frame and send back result
                            processed_frame = await asyncio.get_event_loop().run_in_executor(
                                None, self.process_frame, frame_data, prompt_data
                            )
                            if processed_frame:
                                await ws.send_str(f"processed_frame:{processed_frame}")
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print(f"WebSocket error: {ws.exception()}")
            finally:
                request.app['websockets'].discard(ws)
            return ws

        async def on_shutdown(app):
            for ws in set(app['websockets']):
                await ws.close(code=aiohttp.WSCloseCode.GOING_AWAY, 
                             message='Server shutdown')

        app = web.Application()
        app['websockets'] = weakref.WeakSet()
        
        app.router.add_get("/", handle_index)
        app.router.add_get("/ws", websocket_handler)
        app.on_shutdown.append(on_shutdown)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', 7860)
        await site.start()
        
        # Keep the server running
        while True:
            await asyncio.sleep(3600)