"""
JalVaani API — centralised configuration.

All values can be overridden via environment variables or a .env file.
Copy .env.example → .env and adjust before running in production.
"""
from typing import List

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings  # pydantic v1 fallback


class Settings(BaseSettings):
    # ── CORS ─────────────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins.
    # Set to "*" only during local development; restrict in production.
    cors_origins: str = (
        "http://localhost:5173,"   # Vite dev server
        "http://localhost:4173,"   # Vite preview
        "http://localhost:8000"    # Same-origin (FastAPI serving built UI)
    )

    # ── Rate limiting ─────────────────────────────────────────────────────────
    # Max requests per minute per IP address (in-process sliding window).
    rate_limit_rpm: int = 120

    # ── Static UI ─────────────────────────────────────────────────────────────
    # Absolute or relative path to the built React app (jalvaani_ui/dist).
    # Leave empty to skip — useful when a separate CDN/Nginx serves the UI.
    static_dir: str = ""

    # ── Server ───────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Cache TTLs (seconds) ─────────────────────────────────────────────────
    cache_ttl_national_stats: int = 3600   # 1 hour — data never changes at runtime
    cache_ttl_stations_page: int = 300     # 5 minutes

    @property
    def allowed_origins(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
