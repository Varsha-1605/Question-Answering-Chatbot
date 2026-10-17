import asyncio

from fastapi import APIRouter

from ai_workflow.cache.monitoring import cache_report

router = APIRouter(tags=["ops"])


@router.get("/cache/stats")
async def cache_stats_endpoint() -> dict:
    """Live cache report."""
    return await asyncio.to_thread(cache_report)

