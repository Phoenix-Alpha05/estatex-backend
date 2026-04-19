from fastapi import APIRouter

from app.api.acquisition import router as acquisition_router
from app.api.analysis import router as analysis_router
from app.api.health import router as health_router
from app.api.investment import router as investment_router
from app.api.renovation import router as renovation_router
from app.api.tenant import router as tenant_router
from app.api.workflow import router as workflow_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(tenant_router, tags=["tenant"])
api_router.include_router(investment_router, tags=["investment"])
api_router.include_router(acquisition_router, tags=["acquisition"])
api_router.include_router(renovation_router, tags=["renovation"])
api_router.include_router(workflow_router, tags=["workflow"])
api_router.include_router(analysis_router, tags=["analysis"])
