# Deep Learning & Computer Vision: The Complete Guide

## ML Zoomcamp Module 8 - Comprehensive Notes



### Part 1: The General Deep Learning Workflow

Before touching any code, you must understand the universal lifecycle of a Computer Vision project. This process applies whether you use Keras, PyTorch, or plain NumPy.



* **Problem Definition:**

    * Identify the **Input** (e.g., Image of shape `(224, 224, 3)`).

    * Identify the **Output** (e.g., Vector of probabilities for 10 classes).



* **Data Preparation:**

    * **Collection:** Gathering images.

    * **Splitting:** Standard split is often Train (60%), Validation (20%), Test (20%).

    * **Preprocessing:** Resizing images to a standard size and normalizing pixel values (0-255 $\rightarrow$ 0-1).



* **Model Architecture Design:**

    * **Backbone (Feature Extractor):** The Convolutional layers that "see" the image.

    * **Head (Classifier):** The Dense layers that make the final decision.



* **Training Loop:**

    * **Forward Pass:** Pass image $\rightarrow$ Get Prediction.

    * **Loss Calculation:** Compare Prediction vs. Actual Label (e.g., CrossEntropy).

    * **Backward Pass (Backprop):** Calculate gradients (determines how much each weight contributed to the error).

    * **Optimizer Step:** Update weights to reduce error.



* **Evaluation & Tuning:**

    * Check Validation Accuracy.

    * If overfitting, apply regularization (Dropout, Augmentation).



---



### Part 2: Deep Dive - How CNNs Actually Work

The "Magic" of Deep Learning is Convolution. Standard "Dense" layers fail on images because they destroy spatial relationships (they don't know that pixel 0,0 is next to pixel 0,1).



#### 1. The Convolution Operation (The "Filter")

A Convolutional Layer does not look at the whole image at once. It scans it using a Filter (or Kernel).



* **The Filter:** A small matrix of weights (e.g., $3 \times 3$).

* **The Slide:** The filter slides over the input image.

* **The Math (Dot Product):** At every position, we do element-wise multiplication between the filter values and the image pixels, then sum them up.

* **Result:**

    * If the filter shape matches the image patch, the result is a high number (activation).

    * If they don't match, it's low.



* **Visual Example:**

    * Imagine a filter designed to detect a vertical edge:

    $$

    \begin{bmatrix}

    1 & 0 & -1 \\

    1 & 0 & -1 \\

    1 & 0 & -1

    \end{bmatrix}

    $$

    * If this filter slides over a part of the image with a vertical line, the math yields a high value. This creates a **Feature Map**.



#### 2. Stacking Feature Maps

* **Layer 1:** Detects simple lines and edges.

* **Layer 2:** Combines lines to detect shapes (circles, squares).

* **Layer 3:** Combines shapes to detect objects (eyes, wheels).

* **Note:** We typically use 32, 64, or 128 filters in a single layer. This means the layer outputs a volume of depth 32, 64, or 128 (one "sheet" for each feature).



#### 3. Pooling (Downsampling)

As we go deeper, we don't need pixel-perfect precision; we need "existence" information (e.g., *is there an ear?*).



* **Max Pooling:** We take a window (e.g., $2 \times 2$) and keep only the largest value.

* **Effect:**

    * Reduces image size by 75% (computationally cheaper).

    * Makes the model invariant to translation (it doesn't matter if the cat moves slightly to the left).



#### 4. Vectorization (Flattening)

Before making a final prediction, we must convert the 3D feature maps (Height $\times$ Width $\times$ Channels) into a 1D vector.



* **Global Average Pooling:**

    * Instead of flattening all pixels (which creates millions of parameters), we take the average value of each feature map.

    * Example: If we have 2048 feature maps, we get a vector of size 2048.



---



### Part 3: Implementation - Keras vs. PyTorch

In the course, we primarily use Keras for its simplicity, but understanding the PyTorch "Manual" approach clarifies what is happening under the hood.



#### Approach A: The Keras Way (Declarative)

* **Philosophy:** "Define the structure, and I'll handle the loop."



**1. Data Loading**

```python

from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_ds = train_gen.flow_from_directory('./train', target_size=(150, 150), batch_size=32)



## Model Definition

# Functional API

inputs = keras.Input(shape=(150, 150, 3))

base = Xception(weights='imagenet', include_top=False)(inputs)

base.trainable = False  # Freeze base



# The Head

x = keras.layers.GlobalAveragePooling2D()(base)

x = keras.layers.Dense(128, activation='relu')(x)

outputs = keras.layers.Dense(10, activation='softmax')(x)



model = keras.Model(inputs, outputs)



## Training

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, epochs=10, validation_data=val_ds)



### Approach B: The PyTorch Way (Imperative)

Philosophy: "I will write the loop myself." This gives you total control.

#### Data Loading

import torch

from torchvision import datasets, transforms

from torch.utils.data import DataLoader



transform = transforms.Compose([

    transforms.Resize((150, 150)),

    transforms.ToTensor()

])



train_ds = datasets.ImageFolder('./train', transform=transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

#### model definition

import torch.nn as nn

import torchvision.models as models



class TransferModel(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        # Load Pretrained

        self.base = models.resnet50(pretrained=True)

        

        # Freeze Base (Iterate through params)

        for param in self.base.parameters():

            param.requires_grad = False

            

        # Replace the Head (In ResNet, the head is called 'fc')

        in_features = self.base.fc.in_features

        self.base.fc = nn.Sequential(

            nn.Linear(in_features, 128),

            nn.ReLU(),

            nn.Linear(128, num_classes)

        )



    def forward(self, x):

        return self.base(x)



model = TransferModel(num_classes=10)



### Training Loop (The Manual Process)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



for epoch in range(10):

    for images, labels in train_loader:

        # 1. Forward Pass

        outputs = model(images)

        loss = criterion(outputs, labels)

        

        # 2. Backward Pass

        optimizer.zero_grad() # Clear old gradients

        loss.backward()       # Calculate new gradients

        optimizer.step()      # Update weights


```
### Part 4: Advanced Tuning & "Tricks of the Trade"



#### 1. When to add more layers?

You have your GlobalPooling vector. Should you go straight to the output?



* **Direct (Pooling $\rightarrow$ Output):** Best for small datasets to prevent overfitting.

* **Intermediate (Pooling $\rightarrow$ Dense $\rightarrow$ Output):** Use this when the tasks are very different.

    * *Example:* The base model was trained on dogs/cats (ImageNet), but you are classifying X-rays. You need the extra layer to "re-learn" the features into medical context.



#### 2. Learning Rate Decay

A static Learning Rate (LR) is often not enough.



* **Warmup:** Start small, grow big.

* **Decay:** Start big (e.g., 0.01) to move fast, then shrink (0.001 $\rightarrow$ 0.0001) to fine-tune as you get closer to the solution.



#### 3. Checkpointing

* **Keras:** `ModelCheckpoint(save_best_only=True)`

* **PyTorch:** You must manually check accuracy inside the loop:

    ```python

    if current_val_acc > best_acc:

        torch.save(model.state_dict(), 'best_model.pth')

    ```

### Final Verification Checklist



* **Input Shape:** Did you resize your images? (224x224 or 299x299 depending on the model).

* **Normalization:** Did you scale pixels?

    * *Keras:* `preprocess_input` handles this.

    * *PyTorch:* Usually divide by 255 and normalize with mean/std.

* **Freezing:** Did you remember to freeze the base model layers before the first round of training?

* **Activation:**

    * *Binary Classification?* $\rightarrow$ Sigmoid (1 unit).

    * *Multi-class?* $\rightarrow$ Softmax (N units).




