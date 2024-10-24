# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import asyncio
import argparse
from aiohttp import web, WSCloseCode
import logging
import weakref
import cv2
import time
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import base64
from typing import List
from nanoowl.tree import Tree
from nanoowl.tree_predictor import TreePredictor
from nanoowl.tree_drawing import draw_tree_output
from nanoowl.owl_predictor import OwlPredictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_encode_engine", type=str)
    parser.add_argument("--image_quality", type=int, default=50)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    IMAGE_QUALITY = args.image_quality

    predictor = TreePredictor(
        owl_predictor=OwlPredictor(
            image_encoder_engine=args.image_encode_engine
        )
    )

    prompt_data = None

    def get_colors(count: int):
        cmap = plt.cm.get_cmap("rainbow", count)
        colors = []
        for i in range(count):
            color = cmap(i)
            color = [int(255 * value) for value in color]
            colors.append(tuple(color))
        return colors

    def base64_to_image(base64_str):
        # Remove data URL prefix if present
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[1]
        # Decode base64 string to bytes
        img_data = base64.b64decode(base64_str)
        # Convert to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def cv2_to_pil(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(image)

    async def handle_index_get(request: web.Request):
        logging.info("handle_index_get")
        return web.FileResponse("./index.html")

    async def websocket_handler(request):
        global prompt_data

        ws = web.WebSocketResponse()
        await ws.prepare(request)
        logging.info("Websocket connected.")
        request.app['websockets'].add(ws)

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    if "prompt:" in msg.data:
                        header, prompt = msg.data.split("prompt:")
                        logging.info("Received prompt: " + prompt)
                        try:
                            tree = Tree.from_prompt(prompt)
                            clip_encodings = predictor.encode_clip_text(tree)
                            owl_encodings = predictor.encode_owl_text(tree)
                            prompt_data = {
                                "tree": tree,
                                "clip_encodings": clip_encodings,
                                "owl_encodings": owl_encodings
                            }
                            logging.info("Set prompt: " + prompt)
                        except Exception as e:
                            print(e)
                    elif "frame:" in msg.data:
                        # Process incoming video frame
                        header, frame_data = msg.data.split("frame:")
                        image = base64_to_image(frame_data)
                        
                        # Process frame with NanoOWL
                        image_pil = cv2_to_pil(image)
                        if prompt_data is not None:
                            detections = predictor.predict(
                                image_pil,
                                tree=prompt_data['tree'],
                                clip_text_encodings=prompt_data['clip_encodings'],
                                owl_text_encodings=prompt_data['owl_encodings']
                            )
                            image = draw_tree_output(image, detections, prompt_data['tree'])
                        
                        # Encode and send processed frame back
                        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])
                        base64_image = base64.b64encode(buffer).decode('utf-8')
                        await ws.send_str(f"processed_frame:{base64_image}")

        finally:
            request.app['websockets'].discard(ws)

        return ws

    async def on_shutdown(app: web.Application):
        for ws in set(app['websockets']):
            await ws.close(code=WSCloseCode.GOING_AWAY, message='Server shutdown')

    logging.basicConfig(level=logging.INFO)
    app = web.Application()
    app['websockets'] = weakref.WeakSet()
    app.router.add_get("/", handle_index_get)
    app.router.add_route("GET", "/ws", websocket_handler)
    app.on_shutdown.append(on_shutdown)
    web.run_app(app, host=args.host, port=args.port)