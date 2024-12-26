from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from model.predict import ModelPredictor
from model.monitor import monitor_prediction_time


router = APIRouter()

# Request body schema
class PredictionRequest(BaseModel):
    text: str

predictor = ModelPredictor("model/svm_model.pkl")

monitor = monitor_prediction_time()

@router.post("/predict/")
@monitor
def predict(request: PredictionRequest):
    """
    Predict the label for the given text input.

    Args:
        request (PredictionRequest): Input text wrapped in a request body.

    Returns:
        dict: Status and predicted label.
    """
    try:
        result = predictor.predict(request.text)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))