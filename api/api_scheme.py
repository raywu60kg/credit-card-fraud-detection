from pydantic import BaseModel


class ModelInput(BaseModel):
    TransactionAmt: float
    ProductCD: int
    card1: int
    C1: float
    C2: float
    C3: float
    C4: float
    C5: float
    C6: float
    C7: float
    C8: float
    C9: float
    C10: float
    C11: float
    C12: float
    C13: float
    C14: float


class ModelOutput(BaseModel):
    prediction: float


class HealthCheckOutput(BaseModel):
    health: bool


class MetricsOutput(BaseModel):
    model_name: str
    log_loss: float
    auc: float
    average_precision: float
