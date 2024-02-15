"""A/B experiment endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.gateway.ab_router import ABRouter, Experiment

router = APIRouter(prefix="/v1/experiments", tags=["experiments"])
_ab_router = ABRouter()


class ExperimentCreate(BaseModel):
    id: str
    variants: list[dict]


@router.get("")
def list_experiments():
    return {
        "experiments": [{"id": e.id, "variants": e.variants} for e in _ab_router.list_experiments()]
    }


@router.post("")
def create_experiment(req: ExperimentCreate):
    exp = Experiment(id=req.id, variants=req.variants)
    _ab_router.add_experiment(exp)
    return {"id": exp.id, "variants": exp.variants}


@router.get("/{experiment_id}/assignment")
def get_assignment(experiment_id: str, user_id: str = "default"):
    try:
        model = _ab_router.get_assignment(experiment_id, user_id)
        return {"experiment_id": experiment_id, "user_id": user_id, "model": model}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
