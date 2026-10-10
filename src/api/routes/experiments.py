"""A/B experiment endpoints.

Experiments are held in process memory: they are lost on restart and are not
shared between workers or replicas.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator

from src.api.deps import require_api_key
from src.gateway.ab_router import ABRouter, Experiment

router = APIRouter(prefix="/v1/experiments", tags=["experiments"])
_ab_router = ABRouter()


class Variant(BaseModel):
    model: str
    traffic_pct: int = Field(ge=0, le=100)


class ExperimentCreate(BaseModel):
    id: str
    variants: list[Variant] = Field(min_length=1)

    @model_validator(mode="after")
    def weights_sum_to_100(self) -> "ExperimentCreate":
        total = sum(v.traffic_pct for v in self.variants)
        if total != 100:
            raise ValueError(f"variant traffic_pct values must sum to 100, got {total}")
        return self


@router.get("")
def list_experiments(api_key: str = Depends(require_api_key)):
    return {
        "experiments": [{"id": e.id, "variants": e.variants} for e in _ab_router.list_experiments()]
    }


@router.post("")
def create_experiment(req: ExperimentCreate, api_key: str = Depends(require_api_key)):
    exp = Experiment(id=req.id, variants=[v.model_dump() for v in req.variants])
    _ab_router.add_experiment(exp)
    return {"id": exp.id, "variants": exp.variants}


@router.get("/{experiment_id}/assignment")
def get_assignment(
    experiment_id: str, user_id: str = "default", api_key: str = Depends(require_api_key)
):
    try:
        model = _ab_router.get_assignment(experiment_id, user_id)
        return {"experiment_id": experiment_id, "user_id": user_id, "model": model}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
