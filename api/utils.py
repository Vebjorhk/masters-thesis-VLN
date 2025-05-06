import json
import cv2
from fastapi.responses import JSONResponse
import base64

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def get_path_mapping(data):
    path_dict = {}
    for p in data:
        path_dict[p["path_id"]] = p
    return path_dict

def convert_image_to_response(image, move_possible):
    #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, img_encoded = cv2.imencode('.png', image)
    img_bytes = img_encoded.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')  # Convert image to base64
    return JSONResponse(content={"image": img_base64, "move_possible": move_possible})


