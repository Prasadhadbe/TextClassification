# SVM Text Classification API

This repository contains a simple FastAPI-based API for performing text classification using a pre-trained Support Vector Machine (SVM) model.

## Features

- **Text Classification:** Classify text into predefined categories.
- **Prediction Probability:**  Provides the probability of the predicted class.
- **Performance Monitoring:** Monitors the time taken for each prediction.

## Project Structure

```
├── app
│   ├── routes.py
│   └── main.py
├── model
│   ├── train.py
│   ├── predict.py
│   ├── svm_model.pkl
│   └── monitor.py
├── venv
├── .gitignore
├── dataset.xlsx
├── requirements.txt


```

- **app/main.py:**  Main FastAPI application file.
- **app/routes.py:**  Defines the API routes and handles requests.
- **model/predict.py:** Contains the `ModelPredictor` class for loading the model and making predictions.
- **model/train.py:**  Provides a function to train the SVM model and save it.
- **model/monitor.py:**  Includes a decorator to monitor prediction time.
- **requirements.txt:** Lists the project dependencies.
- **.gitignore:** Specifies files and folders to be ignored by Git.
- **svm_model.pkl:**  The trained SVM model file. (Not included in the repository, generate it using `train.py`)
- **dataset.xlsx:** The Excel file containing the training data with 'text' and 'label' columns.
- **venv:** The virtual environment directory containing Python packages.
- **README.md:** The main documentation file explaining the project, its structure, installation, and usage.


## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/svm-text-classification-api.git 
   ```

2. **Navigate to the project directory:**
   ```bash
   cd svm-text-classification-api
   ```

3. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   ```

4. **Activate the virtual environment:**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source venv/bin/activate
     ```

5. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Training the Model

1. **Prepare your dataset:** You need a dataset in Excel format (.xlsx) with two columns: "text" and "label".
2. **Update the `train.py` script:** Modify the `data_path` variable in `train.py` to point to your dataset.
3. **Run the training script:**
   ```bash
   python model/train.py
   ```
   This will train the SVM model and save it as `svm_model.pkl`.

## Running the API

1. **Start the FastAPI server:**
   ```bash
   uvicorn app.main:app --reload
   ```

## Making Predictions

- **Send a POST request to `/predict/` with the following JSON payload:**

  ```json
  {
    "text": "This is an example text for classification."
  }
  ```

- **The API will respond with the predicted class and probability:**

  ```json
  {
    "status": "success",
    "data": {
      "prediction": "class_label",
      "probability": [0.2, 0.8] 
    }
  }
  ```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License.
