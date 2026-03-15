import logging

import uvicorn


class AccessLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "GET /jobs/" not in message


def run_server() -> None:
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.setLevel(logging.INFO)
    access_logger.filters.clear()
    access_logger.addFilter(AccessLogFilter())

    print("ANPR server running at http://localhost:8000")

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        access_log=True,
        log_level="info",
    )
