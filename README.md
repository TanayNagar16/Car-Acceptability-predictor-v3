# Car Acceptability Predictor

## Overview

This project implements a machine learning model to predict the acceptability of a car based on its features. A user-friendly web application, built using the Streamlit library in Python, provides an interactive interface for users to input car specifications and receive a prediction on its acceptability.

## Dataset

The car evaluation dataset utilized in this project is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/19/car+evaluation). This dataset contains information on various attributes of cars and their corresponding acceptability levels.

## Machine Learning Model

A *Logistic Regression* model was chosen and trained for this classification task. Logistic Regression is well-suited for predicting categorical outcomes like car acceptability.

## How to Run the Project

Follow these steps to run the project on your local machine:

1.  *Prerequisites:*
    * Python 3.x installed on your system. You can download it from [https://www.python.org/downloads/](https://www.python.org/downloads/).
    * pip (Python package installer), which usually comes bundled with Python.

2.  *Installation:*
    Open your terminal or command prompt and navigate to the directory where you have saved the project files (Car Evaluation 1.py, App 1.py, car.data). Then, install the necessary Python libraries using pip:
    bash
    pip install pandas scikit-learn streamlit matplotlib seaborn
    

3.  *Run the Model Training Script:*
    Execute the following command to run the script that trains the machine learning model and saves the trained model and encoders to .pkl files:
    bash
    python "Car Evaluation 1.py"
    
    Wait for the script to finish. You should see output indicating that the model and encoders have been saved as car_evaluation_model.pkl and car_evaluation_encoders.pkl.

4.  *Run the Streamlit Web Application:*
    Once the training script has completed, run the Streamlit web application using the following command:
    bash
    streamlit run "App 1.py"
    
    This command will start the Streamlit development server and automatically open the application in your default web browser. You can usually access it at http://localhost:8501.

## Demo Video

[Link to my Demo Video](https://youtu.be/_xf2m-75oKY)

## Troubleshooting - Potential Antivirus Interference

During the execution of the Car Evaluation 1.py script, some antivirus software might interfere with the process of saving the trained model (car_evaluation_model.pkl) and encoders (car_evaluation_encoders.pkl). If you encounter issues such as FileNotFoundError when running the Streamlit app or other errors during the training script, you can try the following temporary step:

1.  *Temporarily Disable Your Antivirus Software:* Refer to your antivirus software's documentation for instructions on how to temporarily disable it. For Windows Security, the steps were outlined earlier.

2.  **Run the Car Evaluation 1.py script again.**

3.  **After verifying that the .pkl files are created, remember to re-enable your antivirus software.**

For a more permanent solution, consider adding an exclusion or allowed app rule in your antivirus settings for your project directory or for Python.

## Author

Tanay Nagar
BCA 1st Year