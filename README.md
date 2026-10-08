# 🌱 Plant Disease Detection

A deep learning-based plant disease detection project that uses a Convolutional Neural Network (CNN) to classify plant leaf images into disease categories.

The project includes model training and testing notebooks, a trained Keras model, and a Streamlit application for making predictions through a user-friendly interface.

## ✨ Features

- 🌿 Plant disease classification from leaf images
- 🧠 CNN-based deep learning model
- 📓 Jupyter notebooks for training and testing
- 💻 Streamlit web application
- 🗂️ JSON file containing class names
- 📊 Training history saved in JSON format
- 🖼️ Image-based prediction interface

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow
- Jupyter Notebook

## 📁 Project Structure

```text
Plant-Disease-Detection/
│
├── main.py                       # Streamlit application
├── trained_plant_model2.keras    # Trained CNN model
├── Train_plant_disease.ipynb     # Model training notebook
├── Test_Plant_Disease.ipynb      # Model testing notebook
├── class_names.json              # Class/disease names
├── training_hist.json            # Training history
├── home_page.jpeg                # Application image
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── .gitignore                    # Git ignored files
```

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/nischal-027/Plant-Disease-Detection.git
cd Plant-Disease-Detection
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Or, if you already have a Python virtual environment, activate that environment.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

The project uses Streamlit for the web interface.

Run:

```bash
streamlit run main.py
```

After starting Streamlit, open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

Upload a plant leaf image through the application to obtain a disease prediction.

## 🧠 Model

The trained model is stored as:

```text
trained_plant_model2.keras
```

The model is loaded by the application and used to classify uploaded plant images.

The predicted class names are stored separately in:

```text
class_names.json
```

## 📓 Notebooks

### `Train_plant_disease.ipynb`

Contains the workflow used for training the plant disease classification model.

### `Test_Plant_Disease.ipynb`

Contains testing/evaluation and prediction-related work.

## 📊 Training History

Training information is stored in:

```text
training_hist.json
```

This file can be used to inspect the model's training history and visualize metrics such as accuracy and loss when applicable.

## 🔮 Future Improvements

Possible improvements include:

- Improve model accuracy and generalization
- Add more plant species and disease classes
- Add confidence scores and prediction explanations
- Improve the Streamlit user interface
- Deploy the application online
- Add more robust image preprocessing
- Add model performance visualizations
- Use additional validation and test data

## ⚠️ Disclaimer

This project is intended for educational and demonstration purposes. Predictions from a machine learning model should not be treated as a substitute for professional agricultural diagnosis.

## 👨‍💻 Author

**Nischal**

GitHub: [nischal-027](https://github.com/nischal-027)

## 📄 License

A license has not been specified for this repository yet.
If you plan to distribute or reuse the project, consider adding an appropriate open-source license.
