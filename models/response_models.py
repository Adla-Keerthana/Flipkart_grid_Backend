from pydantic import BaseModel


class LabelResponse(BaseModel):
    label: str


class ExpiryResponse(BaseModel):
    expiry_date: str


class FreshnessResponse(BaseModel):
    freshness_score: float