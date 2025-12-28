import importlib.metadata
import os

from fastapi import FastAPI
from loguru import logger
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as OTLPGrpcSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPHttpSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import DEPLOYMENT_ENVIRONMENT, SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_default_tracer_provider() -> bool:
    # Avoid clobbering externally-configured tracing (e.g. `opentelemetry-instrument`).
    return trace.get_tracer_provider().__class__.__name__ == "ProxyTracerProvider"


def _service_version() -> str | None:
    try:
        return importlib.metadata.version("finrag")
    except Exception:
        return None


def setup_opentelemetry(app: FastAPI) -> None:
    """
    Configure OpenTelemetry tracing for the service and instrument FastAPI.

    Enablement:
      - `FINRAG_OTEL_ENABLED=true|false` (default: true if `OTEL_EXPORTER_OTLP_ENDPOINT` is set)
    """

    enabled_default = bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))
    if not _env_bool("FINRAG_OTEL_ENABLED", default=enabled_default):
        logger.info("OpenTelemetry disabled (FINRAG_OTEL_ENABLED=false).")
        return

    if not _is_default_tracer_provider():
        logger.info("OpenTelemetry already configured; skipping finrag setup.")
        FastAPIInstrumentor.instrument_app(app)
        return

    resource = Resource.create(
        {
            SERVICE_NAME: (os.getenv("OTEL_SERVICE_NAME") or "finrag").strip(),
            **({SERVICE_VERSION: v} if (v := _service_version()) else {}),
            **(
                {DEPLOYMENT_ENVIRONMENT: env}
                if (env := (os.getenv("OTEL_DEPLOYMENT_ENVIRONMENT") or "").strip())
                else {}
            ),
        }
    )

    provider = TracerProvider(resource=resource)

    protocol = (os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL") or "http/protobuf").strip().lower()
    if protocol in {"grpc", "grpc/protobuf"}:
        exporter = OTLPGrpcSpanExporter()
    else:
        exporter = OTLPHttpSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))

    if _env_bool("FINRAG_OTEL_CONSOLE", default=False):
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
    logger.info("OpenTelemetry enabled (protocol={}).", protocol)
