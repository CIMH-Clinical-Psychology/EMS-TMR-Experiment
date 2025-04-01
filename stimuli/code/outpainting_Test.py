#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 09:04:47 2025

@author: simon
"""

import requests

def classify_image(deployment_token, deployment_id, image_path):
    url = "https://systemoo.abacus.ai/api/classifyImage"
    headers = {
        "Authorization": f"Bearer {deployment_token}"
    }
    files = {
        "image": open(image_path, "rb")
    }
    data = {
        "deploymentId": deployment_id
    }
    response = requests.post(url, headers=headers, files=files, data=data)
    return response.json()

# Example usage
deployment_token = "<your_deployment_token>"
deployment_id = "<your_deployment_id>"
image_path = "path/to/your/image.jpg"
result = classify_image(deployment_token, deployment_id, image_path)
print(result)
