# Skin Condition Detection Application

This application uses deep learning to detect different skin conditions.

## Installation Guide

Follow these steps to set up and run the application on your local machine.

## 1. Clone the Repository

First, clone this repository to your local machine using the following command:

```bash
git clone https://github.com/MaryamAshraff2/multi-skin-condition-classifier.git
cd multi-skin-condition-classifier
```

## 2. Download Model Weights

The application requires pre-trained model weights to function properly. Please download the weights from the Google Drive link below and:

https://drive.google.com/drive/folders/1B0JBkjw99d5IrGcHfEfTLDiNndXbYnst?usp=sharing 

After downloading, save it in the modals folder in the project root directory.

## 3. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

## 4. Install Requirements
```bash
pip install -r requirements.txt
```

## 5. Run the Application
```bash
python -m streamlit run app.py
```


Access the application at:
```bash
http://localhost:8501
```
