import modal
import os

stub = modal.Stub("nanoowl-app")
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

@stub.cls(
    image=image,
    gpu="t4",
    volumes={ENGINE_DIR: volume},
    memory=32768,
    allow_concurrent_inputs=True,
    cpu=4,
    container_idle_timeout=300
)
class NanoOwl:
    def __init__(self):
        self.is_initialized = False
        self.prompt_data = None

    @modal.enter()
    def initialize(self):
        """Initialize the model and infrastructure"""
        import os
        import subprocess
        from nanoowl.owl_predictor import OwlPredictor
        from nanoowl.tree_predictor import TreePredictor
        
        print("Starting initialization...")
        
        self.engine_path = f"{ENGINE_DIR}/owl_image_encoder_patch32.engine"
        
        if not os.path.exists(self.engine_path):
            print("Building TensorRT engine...")
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
        self.is_initialized = True
        print("Initialization complete!")

    def get_loading_html(self):
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta http-equiv="refresh" content="5">
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

    def process_frame(self, frame_data: str):
        import base64
        import cv2
        import numpy as np
        from PIL import Image
        from nanoowl.tree import Tree
        from nanoowl.tree_drawing import draw_tree_output
        
        try:
            # Decode base64 image
            if 'base64,' in frame_data:
                frame_data = frame_data.split('base64,')[1]
            img_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert to PIL Image
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if self.prompt_data is not None:
                detections = self.predictor.predict(
                    image_pil,
                    tree=self.prompt_data['tree'],
                    clip_text_encodings=self.prompt_data['clip_encodings'],
                    owl_text_encodings=self.prompt_data['owl_encodings']
                )
                image = draw_tree_output(image, detections, self.prompt_data['tree'])
            
            # Encode processed image
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 50])
            processed_frame = base64.b64encode(buffer).decode('utf-8')
            
            return processed_frame
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None

    @modal.asgi_app()
    def app(self):
        from fastapi import FastAPI, WebSocket
        from fastapi.responses import HTMLResponse, FileResponse
        import logging
        import weakref
        from nanoowl.tree import Tree
        
        app = FastAPI()
        app.state.websockets = weakref.WeakSet()
        
        @app.get("/")
        async def root():
            if not self.is_initialized:
                return HTMLResponse(self.get_loading_html())
            return FileResponse("/root/nanoowl/examples/tree_demo/index.html")
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            app.state.websockets.add(websocket)
            
            try:
                async for message in websocket.iter_text():
                    if "prompt:" in message:
                        try:
                            prompt = message.split("prompt:")[1]
                            tree = Tree.from_prompt(prompt)
                            clip_encodings = self.predictor.encode_clip_text(tree)
                            owl_encodings = self.predictor.encode_owl_text(tree)
                            self.prompt_data = {
                                "tree": tree,
                                "clip_encodings": clip_encodings,
                                "owl_encodings": owl_encodings
                            }
                            await websocket.send_text("status:Prompt updated")
                        except Exception as e:
                            print(f"Error processing prompt: {e}")
                            await websocket.send_text(f"error:Failed to process prompt: {str(e)}")
                    
                    elif "frame:" in message:
                        frame_data = message.split("frame:")[1]
                        processed_frame = self.process_frame(frame_data)
                        if processed_frame:
                            await websocket.send_text(f"processed_frame:{processed_frame}")
            
            except Exception as e:
                print(f"WebSocket error: {e}")
            finally:
                app.state.websockets.remove(websocket)
                await websocket.close()

        @app.on_event("shutdown")
        async def shutdown():
            for ws in app.state.websockets:
                await ws.close()

        return app

# For development
if __name__ == "__main__":
    stub.serve()