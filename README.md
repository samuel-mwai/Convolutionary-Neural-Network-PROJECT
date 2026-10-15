# Bag Classification using Convolutional Neural Networks (CNN)

This repository contains a **Convolutional Neural Network (CNN)** implemented in **TensorFlow/Keras** to classify synthetic images of bags into three distinct categories. The model includes early stopping and model checkpointing to ensure optimal training performance.

## 🛠️ Tech Stack
* **Framework:** TensorFlow (Keras)
* **Libraries:** NumPy, Matplotlib
* **Environment:** Python 3.12, Jupyter/Kaggle Notebook

## 📊 Dataset & Classes
The project uses a synthetic dataset of garbage, paper, and plastic bags:
* **Total Files:** 15,000 images
* **Training Set:** 10,500 images (70%)
* **Validation Set:** 4,500 images (30%)

### Target Classes
1. `Garbage Bag Images`
2. `Paper Bag Images`
3. `Plastic Bag Images`

---

## 🧠 Model Architecture
The network processes input images resized to **180 × 180 pixels** across 3 color channels (RGB).

| Layer | Type | Specifications |
| :--- | :--- | :--- |
| 1 | **Input** | 180 × 180 × 3 image dimensions |
| 2 | **Rescaling** | Normalizes pixel values from [0, 255] to [0, 1] |
| 3 | **Conv2D** | 32 filters, 3 × 3 kernel, ReLU activation |
| 4 | **MaxPooling2D** | 2 × 2 pool size |
| 5 | **Conv2D** | 32 filters, 3 × 3 kernel, ReLU activation |
| 6 | **MaxPooling2D** | 2 × 2 pool size |
| 7 | **Conv2D** | 32 filters, 3 × 3 kernel, ReLU activation |
| 8 | **MaxPooling2D** | 2 × 2 pool size |
| 9 | **Flatten** | Transforms 3D matrix to 1D vector |
| 10 | **Dense** | 128 units, ReLU activation |
| 11 | **Dense** | 64 units, ReLU activation |
| 12 | **Dense (Output)**| 3 units, Softmax activation |

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com
cd bag-classification-cnn
```

### 2. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install tensorflow numpy matplotlib pillow ipywidgets
```

### 3. Dataset Configuration
Update the `data_dir` variable in the notebook to point to your local dataset directory:
```python
data_dir = "path/to/your/dataset"
```

---

## 💡 Training & Optimization
The model compiles using the **Adam optimizer** and **Sparse Categorical Crossentropy** loss. 

To prevent overfitting, the following callbacks are implemented during the maximum 50-epoch training cycle:
* **ModelCheckpoint:** Automatically saves only the best-performing model weights to `best_model.h5` based on validation loss.
* **EarlyStopping:** Stops training if the `val_loss` does not improve for 5 consecutive epochs, restoring the best historical weights.

---

## 🔮 Interactive Interface
The project contains an embedded **IPython FileUpload widget**. You can run the notebook cell to upload your own local `.jpg` or `.jpeg` images and test the trained model's classification confidence in real-time.

---

## 👨‍💻 Author

**Samuel Mwai**  
Data Science & Analytics | Machine Learning | Data Analysis | Data Visualization  
* **GitHub:** [samuel-mwai](https://github.com/samuel-mwai)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
