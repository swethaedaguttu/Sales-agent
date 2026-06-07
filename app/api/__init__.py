from app.api.catalog import router as catalog_router
from app.api.chat import router as chat_router
from app.api.flags import router as flags_router
from app.api.health import router as health_router

__all__ = ["catalog_router", "chat_router", "flags_router", "health_router"]
