# 📘 Module 9 Master Notes: Serverless Deep Learning

## 1. The Core Concept: Why Serverless?

### The Problem with Traditional Deployment
In the past, to run a deep learning model, you had to rent a server (like an **AWS EC2** instance).
* **Cost:** You pay for the server 24/7, even if no one is using your app at 3 AM.
* **Maintenance:** You have to manage Linux updates, security patches, and scaling (adding more servers if traffic spikes).

### The Solution: Serverless (AWS Lambda)
"Serverless" doesn't mean there are no servers; it means **you don't manage them**. You just give your code to a cloud provider (AWS), and they run it only when a user asks for a prediction.
* **Pay-per-use:** You are charged only for the milliseconds your code runs. If no one calls your API, you pay $0.
* **Auto-scaling:** If 1,000 users hit your endpoint at once, AWS instantly spins up 1,000 parallel copies of your function.

---

## 2. The Model: Moving from Keras to ONNX

You cannot just upload your `model.h5` (Keras/TensorFlow) file to AWS Lambda easily.
* **Problem 1: Size.** TensorFlow is huge (500MB+). AWS Lambda containers must be lean.
* **Problem 2: Speed.** TensorFlow is optimized for *training* (finding patterns), not *inference* (making quick predictions).

### The Fix: ONNX (Open Neural Network Exchange)
ONNX is a universal format for machine learning models. It strips away all the "training" bloat and leaves only the math needed for prediction.

**Key Libraries:**
* `tf2onnx`: Converts your TensorFlow/Keras model to `.onnx`.
* `onnxruntime`: A lightweight engine (C++ wrapped in Python) to run the model.

### Code: Converting Keras to ONNX
*(Run this in your Jupyter Notebook during training)*

```python
import tf2onnx
import tensorflow as tf

# 1. Load your trained Keras model
model = tf.keras.models.load_model('my_model.h5')

# 2. Define the input signature (Crucial: tells ONNX what the data looks like)
# (Batch size: None, Size: 150x150, Channels: 3 for RGB)
spec = (tf.TensorSpec((None, 150, 150, 3), tf.float32, name="input"),)

# 3. Convert and Save
output_path = "model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
```
## 3. The Code: Writing the Lambda Handler

AWS Lambda requires a specific function structure.  
It doesn't run a "script" from top to bottom; it waits for an **Event** (the trigger) and runs a **Handler Function**.

---

### 🔑 Key Concept: The `event` Object

When a user sends a request (e.g., via **API Gateway**), the data comes in as a JSON dictionary called `event`.

**Example:**

```json
{"url": "http://image.com/pic.jpg"}
```
**Your code receives**:event = {'url': 'http://image.com/pic.jpg'}
### The Complete lambda_function.py
This is the standard template for Computer Vision on Lambda.

## 📝 PyTorch → NumPy/PIL Conversions (Lightweight Preprocessing)

During deployment preparation, I replaced several PyTorch functions with pure NumPy/PIL equivalents to make the preprocessing pipeline lighter and avoid heavy dependencies.  
This ensures the ONNX model can run smoothly in AWS Lambda or Docker without requiring `torch` or `torchvision`.

### 🔄 Conversion Cheatsheet

| Task                        | PyTorch Version                          | NumPy/PIL Version (Final)                  |
|-----------------------------|------------------------------------------|--------------------------------------------|
| Convert image to tensor     | `torch.tensor(np.array(img)).float()`<br>`transforms.ToTensor()(img)` | `np.array(img, dtype='float32')` |
| Normalize with mean & std   | `transforms.Normalize(mean, std)(x)`     | `(x - np.array(MEAN, dtype='float32')) / np.array(STD, dtype='float32')` |
| Add batch dimension         | `x.unsqueeze(0)`                         | `np.expand_dims(x, axis=0)` |
| Resize image                | `transforms.Resize(target_size)(img)`    | `img.resize(target_size, Image.NEAREST)` |
| Permute dimensions          | `x.permute(2,0,1)`                       | `x.transpose((2,0,1))` |
| Force float type            | `x.float()`                              | `x.astype(np.float32)` |

---

### ✅ Why This Matters
- **No PyTorch dependency** → smaller Docker/Lambda package size.  
- **Lightweight preprocessing** → faster cold starts in AWS Lambda.  
- **Reproducibility** → same math as PyTorch, but implemented manually with NumPy.  

These swaps ensure the ONNX model runs consistently across environments without requiring the full PyTorch stack.

```python
import numpy as np
import onnxruntime as ort
from PIL import Image
from urllib import request
from io import BytesIO
import os

# ---------------------------------------------------------
# PRO TIP: Initialize the model OUTSIDE the handler.
# Why? AWS keeps the container "warm" for a few minutes.
# If the model is loaded here, it stays in memory for the next request,
# making subsequent predictions much faster.
# ---------------------------------------------------------

# Opoku ML Lesson: Always use environment variables or absolute paths
model_path = os.getenv('MODEL_PATH', 'model.onnx')
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

# 1. Helper: Download Image
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

# 2. Helper: Preprocessing (MUST match your training code EXACTLY)
def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess(url):
    img = download_image(url)
    img = prepare_image(img, (150, 150)) # Example size
    x = np.array(img, dtype='float32')
    x = x / 255.0  # Rescale (0-1)
    x = np.expand_dims(x, axis=0) # Add batch dimension: (1, 150, 150, 3)
    return x

# 3. The Inference Logic
def predict(url):
    X = preprocess(url)
    inputs = {input_name: X}
    output = session.run(None, inputs)[0]
    return float(output[0][0])

# 4. The Lambda Handler (The Entry Point)
def lambda_handler(event, context):
    url = event.get('url')
    result = predict(url)
    return {
        'prediction': result
    }
```
## 4. The Infrastructure: Docker & Containerization

AWS Lambda runs on **Amazon Linux**.  
If you build your project on **Windows** or **Mac**, it might not run there.  
**Docker** solves this by creating a "portable box" that matches the production environment exactly.

---

### 🐳 The Dockerfile Breakdown

This uses a **Multi-Stage Build** (a pro technique to keep images small).

```python
# Base Image: Use a Lambda-compatible image (Amazon Linux 2 + Python)
FROM public.ecr.aws/lambda/python:3.10

# ---------------------------------------------------
# OPOKU ML LESSON #1: PYTHON VERSION MISMATCH
# Your laptop might be Python 3.12, but this image is 3.10.
# Ensure your pyproject.toml allows 3.10, or pip/uv will fail.
# ---------------------------------------------------

# Install Tools (using uv for speed, as you learned)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency file
COPY pyproject.toml .

# ---------------------------------------------------
# OPOKU ML LESSON #2: OS DEPENDENCIES (GLIBC)
# 'onnxruntime' depends on C++ system libraries.
# The latest version might require a newer Linux than Lambda has.
# FIX: In pyproject.toml, pin older versions if needed (e.g., onnxruntime<1.17)
# ---------------------------------------------------
RUN uv pip install --system -r pyproject.toml

# Copy the Model and Code
# ---------------------------------------------------
# OPOKU ML LESSON #3: THE BLACK BOX
# Docker sees NOTHING from your computer unless you COPY it.
# If you don't copy 'model.onnx', the container can't find it.
# ---------------------------------------------------
COPY model.onnx .
COPY lambda_function.py .

# Tell Lambda where to start
# Format: filename.function_name
CMD [ "lambda_function.lambda_handler" ]
```
#### 🐳 The .dockerignore (Crucial for Speed)

**Opoku ML Lesson #4:**  
Without this, Docker tries to upload your entire **80MB `.data` file** and your **`.venv` folder** to the build server.  
This causes **slow builds** and even **"Out of Memory" crashes**.

---

## 📂 File: `.dockerignore`

```text
.venv/
__pycache__/
.git
*.data
.ipynb_checkpoints
```
## 🚀 Deployment: AWS ECR & Lambda

Once your Docker image works locally (`docker run ...`), you must push it to the cloud.

---

### Step 1: 🗂️ ECR (Elastic Container Registry)

Think of this as **"GitHub for Docker Images."**  
You store your built images here.

**Commands:**

```bash
# Login
aws ecr get-login-password ... | docker login ...

# Create Repository
aws ecr create-repository --repository-name my-model

# Tag Image
docker tag my-model:latest ...

# Push Image
docker push ...
```
### Step 2: ⚡ Create Lambda Function

1. Go to **AWS Lambda Console**.  
2. Click **Create Function → Container Image**.  
3. Browse **ECR** and select the image you just pushed.  

**Important Settings:**
- **Memory:** Increase to `1024MB` or `2048MB` (Default `128MB` is too small for ML).  
- **Timeout:** Increase to `30 seconds` (Default `3s` is too short for model loading).  

### Step 3: 🌐 API Gateway (Exposing to the World)

Lambda is **internal**. To let users call it from a website, you need a "Door."

1. Go to **API Gateway**.  
2. Create a **REST API**.  
3. Create a **Resource** (e.g., `/predict`) and a **Method** (`POST`).  
4. Link the Method to your **Lambda Function**.  
5. Deploy the API to a **Stage** (e.g., `prod`).  

✅ You now have a URL:  (https://xyz.execute-api.us-east-1.amazonaws.com/prod/predict)

## 6. 🧠 Opoku ML Pro-Tips Summary (For Revision)

| **Issue**          | **Symptom**                                      | **The Fix**                                                                 |
|---------------------|--------------------------------------------------|------------------------------------------------------------------------------|
| **Python Mismatch** | `uv` fails to resolve dependencies during build. | Check `pyproject.toml` `requires-python`. Match the base image (usually 3.9 or 3.10). |
| **OS Conflict**     | `ImportError` or GLIBC not found when running.   | Pin older library versions (e.g., `numpy<2.0`, `onnxruntime<1.17`).          |
| **Missing File**    | `NoSuchFile: model.onnx` inside container.       | 1. Ensure `COPY` is in Dockerfile.<br>2. Use `find / -name *.onnx` to debug path. |
| **Slow Build**      | `"Sending build context"` takes forever.         | Add `.dockerignore` to exclude `.venv` and datasets.                         |
| **Hidden Path**     | Code crashes because model is in `/app` but Lambda is in `/var/task`. | Use `os.getenv` or absolute paths. Do not assume current directory.          |

---

### 📖 Why These Issues Happen

These problems are common when deploying ML models with **Docker + AWS Lambda** because the environments differ from your local machine:

- **Python Mismatch:** Your laptop may run a newer Python version than Lambda’s base image. Always align versions.  
- **OS Conflicts:** Lambda uses Amazon Linux, which may not support the latest library builds. Pin stable versions.  
- **Missing Files:** Containers are isolated — if you don’t explicitly `COPY` files, they won’t exist inside.  
- **Slow Builds:** Docker uploads the entire folder during build. Large files (like `.venv` or datasets) slow everything down.  
- **Hidden Paths:** Lambda enforces `/var/task` as the working directory, but base images may store files elsewhere. Use absolute paths or environment variables to avoid surprises.  

---






