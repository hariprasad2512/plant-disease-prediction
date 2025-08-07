# Plant Disease Predictor

A Convolutional Neural Network (CNN)–based machine learning model that classifies whether a plant leaf image is healthy or diseased.

---

##  Features
- Image-based plant health classification using CNN.
- Simple user interface powered by **Streamlit** for real-time inference.
- Extensible—easily customizable for other plant disease datasets or fine-tuning.

---

##  Demo

Launch the app and interactively upload leaf images to see real-time disease predictions.

---

##  Getting Started

### Prerequisites
- Python 3.7 or newer
- Internet access (to install dependencies)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/hariprasad2512/plant-disease-prediction.git
cd plant-disease-prediction

# 2. Install required Python packages
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run main.py
```

Once the server initiates, Streamlit will open a browser window (or provide a link) for the interface.

---

##  Project Structure

```
plant-disease-prediction/
├── class_indices.json         # Mapping labels to disease classes
├── main.py                    # Streamlit front-end script
├── Plant_Disease_Prediction_using_CNN.ipynb  # Notebook for training & evaluation
├── requirements.txt           # List of required Python libraries
├── trained_model/             # Directory containing the trained CNN model
└── README.md                  # This documentation file
```

---

##  Dataset

This model uses the [PlantVillage Dataset on Kaggle], which offers labeled leaf images for healthy and diseased plants. To download it:

1. Visit the [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset/data).
2. Download and unzip the dataset into your local `data/` directory.
3. Update paths in the notebook if using a custom dataset layout.

---

##  Usage Example

Once the Streamlit app is running:
1. Upload your leaf image via the UI.
2. The model processes the image and displays:
   - Predicted class (e.g., Healthy, Disease Name)
   - Confidence score or probability

---

##  Training Your Own Model

If you’d like to retrain or fine-tune:

1. Open the notebook: `Plant_Disease_Prediction_using_CNN.ipynb`
2. Prepare your dataset with structure compatible to `class_indices.json`
3. Update model architecture or hyperparameters
4. Retrain the model and export the updated weights to `trained_model/`
5. Adjust `main.py` to point to the new model checkpoint

---

##  Dependencies

Dependencies are listed in `requirements.txt`, including but not limited to:

- `streamlit`
- `tensorflow` or `keras`
- `numpy`
- `opencv-python`
- Any other specified packages

---

##  Contributing

Contributions are most welcome! To contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/YourFeatureName`
3. Make your changes and commit: `git commit -m "Add ..."`
4. Push to your branch: `git push origin feature/YourFeatureName`
5. Open a Pull Request—I'll be happy to review.


---

[PlantVillage Dataset on Kaggle]: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset/data
