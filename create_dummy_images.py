#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 18:12:47 2025

@author: simon
"""

from PIL import Image, ImageDraw, ImageFont
import random
from tqdm import tqdm


n_images = 240 + 120 + 40
categories = ['car', 'dog', 'face', 'flower']


#%%
def create_word_image(word, output_filename=None, width=800, height=400, font_size=80):
    """
    Creates an image with a word on a randomly colored background.

    Args:
        word (str): The word to display on the image
        output_filename (str, optional): Filename to save the image. If None, defaults to '{word}.png'
        width (int): Width of the image in pixels
        height (int): Height of the image in pixels
        font_size (int): Font size for the text

    Returns:
        PIL.Image: The generated image
    """
    # Generate random background color (RGB)
    bg_color = (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )

    # Create a new image with the random background color
    image = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(image)

    # Choose a contrasting text color (simple inverse)
    text_color = (255 - bg_color[0], 255 - bg_color[1], 255 - bg_color[2])

    # Try to load a default font, or use default if not available
    try:
        # Try to use a common font
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # Fall back to default font
        font = ImageFont.load_default()

    # Calculate text size to center it
    text_width, text_height = draw.textsize(word, font=font) if hasattr(draw, 'textsize') else (0, 0)
    if text_width == 0:  # For newer PIL versions
        try:
            text_bbox = draw.textbbox((0, 0), word, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            # If all else fails, make a rough estimate
            text_width, text_height = font_size * len(word) // 2, font_size

    # Calculate position to center the text
    position = ((width - text_width) // 2, (height - text_height) // 2)

    # Draw the text
    draw.text(position, word, fill=text_color, font=font)

    # Save the image if filename is provided
    if output_filename is None:
        output_filename = f"{word}.png"

    image.save(output_filename)
    # print(f"Image saved as: {output_filename}")

    return image



tqdm_loop = tqdm(total=n_images*len(categories), desc='creating image')

for cat in categories:
    for i in range(n_images):
        create_word_image(f'{cat}{i:03d}',
                          output_filename=f'stimuli/{cat}/{cat}{i:03d}.png',
                          width=400,
                          height=400,
                          font_size=80)
        tqdm_loop.update()
