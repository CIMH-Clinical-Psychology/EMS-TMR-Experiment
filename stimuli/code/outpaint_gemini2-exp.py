#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 11:19:47 2025

NOT GIVING GOOD RESULTS CURRENTL :-/

try outpainting with gemini 2 flash experimental

@author: simon
"""
import imageio
import base64
import os
import io
import mimetypes
from google import genai
from google.genai import types
from tqdm import tqdm

def outpaint(image, out_image):
    img_input = imageio.imread(image)
    w = h = max(img_input.shape)

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    files = [
        # Make the file available in local system working directory
        client.files.upload(file=image),
    ]
    model = "gemini-2.0-flash-exp-image-generation"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=files[0].uri,
                    mime_type=files[0].mime_type,
                ),
                types.Part.from_text(text="""make image sharper"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_modalities=[
            "image",
            "text",
        ],
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_CIVIC_INTEGRITY",
                threshold="OFF",  # Off
            ),
        ],
        response_mime_type="text/plain",
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
            continue
        if chunk.candidates[0].content.parts[0].inline_data:
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            img = imageio.imread(io.BytesIO(inline_data.data))
            imageio.imwrite(out_image, img, quality=80)

        else:
            print(chunk.text)

if __name__ == "__main__":

    folder = './stimuli_1/dogs2.0/'
    images = [os.path.join(folder, f)  for f in os.listdir(folder) if f.endswith('.jpg')]
    out_folder = folder + '/gemini2-exp/'
    os.makedirs(out_folder, exist_ok=True)
    for i, image in enumerate(tqdm(images[10:])):
        basename = os.path.basename(image)
        out_image = os.path.join(out_folder, basename[:-4] + '_gemini2exp.jpg')
        outpaint(image, out_image)
        if i>10:
            break
