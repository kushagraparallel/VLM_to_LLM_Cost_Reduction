"""Run the API using host, port, and logging settings from the environment."""

import uvicorn

from app.settings import get_settings


def main() -> None:
    """Start the local development server."""
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.app_env.lower() == "development",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
