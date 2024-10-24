import modal
import os

app = modal.App("nanoowl-app")
volume = modal.Volume.from_name("nanoowl-engines", create_if_missing=True)

ENGINE_DIR = "/root/data"

# Define HTML_CONTENT at module level, before the class
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>NanoOWL Demo</title>
    <style>
        body {
            width: 100%;
            height: 100%;
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f5f5f5;
        }

        #main_container {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            max-width: 90vw;
            width: 800px;
        }

        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 20px;
        }
        
        #prompt_input {
            width: calc(100% - 20px);
            font-size: 18px;
            margin: 16px 0;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 4px;
            transition: border-color 0.3s;
        }

        #prompt_input:focus {
            border-color: #3498db;
            outline: none;
        }

        #camera_image {
            width: 100%;
            border-radius: 4px;
            margin-bottom: 10px;
        }

        .loading-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(255, 255, 255, 0.95);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }

        .loading-spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loading-text {
            margin-top: 20px;
            font-size: 18px;
            color: #333;
        }

        .error-message {
            color: #e74c3c;
            background: #fde8e8;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            text-align: center;
            display: none;
        }

        .status-message {
            color: #2ecc71;
            background: #e8f8f5;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            text-align: center;
            display: none;
        }

        .examples {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 4px;
        }

        .examples h3 {
            margin-top: 0;
            color: #444;
        }

        .examples ul {
            margin: 0;
            padding-left: 20px;
        }

        .examples li {
            margin: 5px 0;
            color: #666;
            cursor: pointer;
        }

        .examples li:hover {
            color: #3498db;
        }
    </style>

    <script type="text/javascript">
        let ws;
        let videoStream;
        let videoElement;
        let canvasElement;
        let canvasContext;
        let reconnectAttempts = 0;
        const MAX_RECONNECT_ATTEMPTS = 5;
        let lastProcessingTime = Date.now();
        const MINIMUM_PROCESSING_INTERVAL = 100; // Minimum 100ms between frames (max 10 fps)
        let processingFrame = false;
        
        function showLoading(message = "Initializing...") {
            const loadingContainer = document.getElementById('loading-container');
            const loadingText = document.getElementById('loading-text');
            loadingText.textContent = message;
            loadingContainer.style.display = 'flex';
        }

        function hideLoading() {
            document.getElementById('loading-container').style.display = 'none';
        }

        function showError(message) {
            const errorDiv = document.getElementById('error-message');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
            setTimeout(() => {
                errorDiv.style.display = 'none';
            }, 5000);
        }

        function showStatus(message) {
            const statusDiv = document.getElementById('status-message');
            statusDiv.textContent = message;
            statusDiv.style.display = 'block';
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 3000);
        }

        async function setupWebcam() {
            try {
                showLoading("Requesting camera access...");
                videoStream = await navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        frameRate: { ideal: 10, max: 15 }
                    } 
                });
                
                videoElement = document.createElement('video');
                videoElement.srcObject = videoStream;
                videoElement.play();
                
                canvasElement = document.createElement('canvas');
                canvasElement.width = 640;
                canvasElement.height = 480;
                canvasContext = canvasElement.getContext('2d', {
                    willReadFrequently: true
                });
                
                videoElement.onplaying = () => {
                    hideLoading();
                    sendFrame();
                };
            } catch (error) {
                console.error('Error accessing webcam:', error);
                showError('Could not access camera. Please ensure camera permissions are granted.');
            }
        }

        async function sendFrame() {
            if (ws && ws.readyState === WebSocket.OPEN && !processingFrame) {
                const currentTime = Date.now();
                if (currentTime - lastProcessingTime >= MINIMUM_PROCESSING_INTERVAL) {
                    processingFrame = true;
                    
                    try {
                        canvasContext.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
                        const frame = canvasElement.toDataURL('image/jpeg', 0.5);
                        ws.send('frame:' + frame);
                        lastProcessingTime = currentTime;
                    } catch (error) {
                        console.error('Error processing frame:', error);
                    } finally {
                        processingFrame = false;
                    }
                }
            }
            requestAnimationFrame(sendFrame);
        }

        async function connectWebSocket() {
            showLoading("Connecting to server...");
            
            return new Promise((resolve, reject) => {
                const wsUrl = "wss://" + location.host + "/ws";
                console.log("Connecting to WebSocket:", wsUrl);
                
                try {
                    ws = new WebSocket(wsUrl);
                    
                    ws.onopen = function() {
                        console.log("Connected to server");
                        reconnectAttempts = 0;
                        hideLoading();
                        resolve(ws);
                    };
                    
                    ws.onmessage = function(event) {
                        if (typeof event.data === 'string') {
                            if (event.data.startsWith('processed_frame:')) {
                                const base64Image = event.data.split('processed_frame:')[1];
                                document.getElementById("camera_image").src = 'data:image/jpeg;base64,' + base64Image;
                            } else if (event.data.startsWith('status:')) {
                                showStatus(event.data.split('status:')[1]);
                            } else if (event.data.startsWith('error:')) {
                                showError(event.data.split('error:')[1]);
                            }
                        }
                    };
                    
                    ws.onerror = function(error) {
                        console.error("WebSocket error:", error);
                        if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
                            reconnectAttempts++;
                            showLoading(`Connection failed. Retrying (${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})...`);
                            setTimeout(connectWebSocket, 2000);
                        } else {
                            showError("Failed to connect to server. Please refresh the page to try again.");
                            reject(error);
                        }
                    };
                    
                    ws.onclose = async function(event) {
                        console.log("WebSocket closed with code:", event.code);
                        if (videoStream) {
                            videoStream.getTracks().forEach(track => track.stop());
                        }
                        
                        if (event.code === 1006 && reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
                            reconnectAttempts++;
                            showLoading(`Connection lost. Retrying (${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})...`);
                            await new Promise(resolve => setTimeout(resolve, 2000));
                            try {
                                await connectWebSocket();
                            } catch (error) {
                                showError("Failed to reconnect. Please refresh the page.");
                            }
                        } else {
                            showError("Connection lost. Please refresh the page to reconnect.");
                        }
                    };
                    
                } catch (error) {
                    console.error("Failed to create WebSocket:", error);
                    reject(error);
                }
            });
        }

        function setExample(prompt) {
            const promptInput = document.getElementById("prompt_input");
            promptInput.value = prompt;
            ws.send("prompt:" + prompt);
        }

        window.onload = async function() {
            try {
                await connectWebSocket();
                await setupWebcam();

                const promptInput = document.getElementById("prompt_input");
                promptInput.oninput = function(event) {
                    console.log("Sending prompt: " + event.target.value);
                    ws.send("prompt:" + event.target.value);
                };
            } catch (error) {
                console.error("Failed to initialize:", error);
                showError("Failed to initialize. Please check your camera and refresh the page.");
            }
        };
    </script>
</head>
<body>
    <div id="loading-container" class="loading-container">
        <div class="loading-spinner"></div>
        <div id="loading-text" class="loading-text">Initializing...</div>
    </div>
    
    <div id="main_container">
        <h1>NanoOWL Demo</h1>
        <img id="camera_image" src="" alt="Camera Feed"/>
        <input id="prompt_input" type="text" placeholder="Enter your prompt here..."/>
        <div id="error-message" class="error-message"></div>
        <div id="status-message" class="status-message"></div>
        
        <div class="examples">
            <h3>Example Prompts:</h3>
            <ul>
                <li onclick="setExample('[a face [a nose, an eye, a mouth]]')">
                    Detect face features: [a face [a nose, an eye, a mouth]]
                </li>
                <li onclick="setExample('[a face (interested, yawning / bored)]')">
                    Classify facial expressions: [a face (interested, yawning / bored)]
                </li>
                <li onclick="setExample('(indoors, outdoors)')">
                    Classify environment: (indoors, outdoors)
                </li>
                <li onclick="setExample('[a person [a hand, a face]]')">
                    Detect person features: [a person [a hand, a face]]
                </li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

# Create image with required setup
image = (modal.Image.from_dockerfile("docker/23-01/Dockerfile")
         .pip_install([
             "torch",
             "transformers",
             "tensorrt", 
             "torch2trt",
             "aiohttp",
             "opencv-python-headless==4.8.0.74",  # Use this specific version
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
            
            # Convert to numpy array and make it writable
            nparr = np.frombuffer(img_data, np.uint8).copy()  # Add .copy() to make writable
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                print("Failed to decode image")
                return None
                
            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image for processing
            pil_img = Image.fromarray(img.copy())  # Add .copy() to make writable
            
            # Process image if prompt data exists
            if self.prompt_data:
                try:
                    tree_output = self.predictor.predict(
                        image=pil_img,
                        tree=self.prompt_data["tree"],
                        clip_text_encodings=self.prompt_data["clip_encodings"],
                        owl_text_encodings=self.prompt_data["owl_encodings"]
                    )
                    
                    # Draw tree output on image
                    pil_img = draw_tree_output(pil_img, tree_output, tree=self.prompt_data["tree"], draw_text=True)
                    
                    # Convert back to numpy array
                    img = np.array(pil_img)
                    
                except Exception as e:
                    print(f"Error in prediction/drawing: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Convert back to BGR for OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Encode processed image to base64
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return img_base64
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()
            return None

    @modal.asgi_app()
    def app(self):
        from fastapi import FastAPI, WebSocket
        from fastapi.responses import HTMLResponse, FileResponse
        from fastapi.middleware.cors import CORSMiddleware
        import logging
        import weakref
        from nanoowl.tree import Tree
        
        app = FastAPI()
        
        # Initialize websockets set at app creation
        if not hasattr(app.state, "websockets"):
            app.state.websockets = set()
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @app.get("/")
        async def root():
            if not self.is_initialized:
                return HTMLResponse(self.get_loading_html())
            return HTMLResponse(HTML_CONTENT)
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            
            try:
                # Store websocket connection
                if not hasattr(app.state, "websockets"):
                    app.state.websockets = set()
                app.state.websockets.add(websocket)
                
                async for message in websocket.iter_text():
                    try:
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
                    except Exception as inner_e:
                        print(f"Error processing message: {inner_e}")
                        import traceback
                        traceback.print_exc()
                        continue  # Continue processing next message instead of breaking
            
            except Exception as e:
                print(f"WebSocket error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Clean up websocket connection
                if hasattr(app.state, "websockets"):
                    app.state.websockets.remove(websocket)
                try:
                    await websocket.close()
                except:
                    pass  # Ignore errors during close

        @app.on_event("startup")
        async def startup():
            app.state.websockets = set()

        @app.on_event("shutdown")
        async def shutdown():
            if hasattr(app.state, "websockets"):
                for ws in app.state.websockets:
                    await ws.close()

        return app