import requests
import base64

with open("test_echo.png", "rb") as f:
    image_bytes = f.read()

clinical_data = {
    "age":50,"bsa":1.7,"sbp":120,"dbp":80,"hr":70
    # other 21 features will default to 0 in Flask
}

payload = {"clinical": clinical_data, "image_base64": base64.b64encode(image_bytes).decode("utf-8")}

response = requests.post("http://127.0.0.1:5000/predict", json=payload)
print(response.json())
