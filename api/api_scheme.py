from pydantic import BaseModel
from typing import List


class ModelInput(BaseModel):
    TransactionAmt: float = 50
    ProductCD: float = 1
    card1: int = 5220
    C1: float = 1
    C2: float = 1
    C3: float = 0
    C4: float = 1
    C6: float = 1
    C7: float = 1
    C8: float = 1
    C9: float = 0
    C10: float = 1
    C11: float = 1
    C12: float = 0
    C13: float = 1


class ModelOutput(BaseModel):
    prediction: List[float]
