from services.model_services.base import BaseModelService, PredictionResult
from services.model_services.fundus_classifier_service import FundusClassifierService
from services.model_services.oct_classifier_service import OCTClassifierService

__all__ = [
	"BaseModelService",
	"PredictionResult",
	"OCTClassifierService",
	"FundusClassifierService",
]
