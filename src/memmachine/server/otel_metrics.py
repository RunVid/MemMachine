"""OpenTelemetry metrics integration for pushing to SigNoz OTEL collector."""

import logging
import os
from typing import Any

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

from memmachine.common.metrics_factory.metrics_factory import MetricsFactory

logger = logging.getLogger(__name__)


def setup_otel_metrics() -> metrics.Meter | None:
    """
    Set up OpenTelemetry metrics to push to SigNoz OTEL collector.

    Environment variables:
    - OTEL_METRICS_ENABLED: "true" to enable OTEL metrics push (default: "false")
    - OTEL_EXPORTER_OTLP_ENDPOINT: SigNoz collector endpoint (default: "http://signoz-otel-collector:4317")
    - OTEL_EXPORTER_OTLP_PROTOCOL: Protocol to use - "grpc" or "http/protobuf" (default: "grpc")
    - OTEL_SERVICE_NAME: Service name (default: "memmachine")
    - OTEL_EXPORT_INTERVAL: Export interval in seconds (default: "15")
    - OTEL_INSECURE: Use insecure connection (default: "true")
    - DEPLOYMENT_ENV: Deployment environment (default: "production")

    Returns:
        Meter instance if enabled, None otherwise
    """
    otel_enabled = os.getenv("OTEL_METRICS_ENABLED", "false").lower() == "true"

    if not otel_enabled:
        logger.info("OpenTelemetry metrics push disabled (OTEL_METRICS_ENABLED=false)")
        return None

    try:
        # Configuration from environment
        endpoint = os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://signoz-otel-collector:4317"
        )
        protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc").lower()
        service_name = os.getenv("OTEL_SERVICE_NAME", "memmachine")
        export_interval = int(os.getenv("OTEL_EXPORT_INTERVAL", "15"))
        insecure = os.getenv("OTEL_INSECURE", "true").lower() == "true"
        deployment_env = os.getenv("DEPLOYMENT_ENV", "production")

        logger.info(
            "Setting up OpenTelemetry metrics push: endpoint=%s, protocol=%s, service=%s, interval=%ds",
            endpoint,
            protocol,
            service_name,
            export_interval,
        )

        # Create resource with service info
        resource_attributes = {
            "service.name": service_name,
            "deployment.environment": deployment_env,
        }

        # Add pod/instance info if available
        pod_name = os.getenv("POD_NAME") or os.getenv("HOSTNAME")
        if pod_name:
            resource_attributes["service.instance.id"] = pod_name

        namespace = os.getenv("POD_NAMESPACE")
        if namespace:
            resource_attributes["k8s.namespace.name"] = namespace

        resource = Resource.create(resource_attributes)

        # Create OTLP exporter based on protocol
        if protocol == "http/protobuf" or protocol == "http":
            try:
                from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                    OTLPMetricExporter,
                )

                # For HTTP protocol, ensure the endpoint has the correct path
                # OTLP HTTP expects /v1/metrics for metrics endpoint
                http_endpoint = endpoint
                if not endpoint.endswith("/v1/metrics"):
                    # Remove trailing slash if present
                    http_endpoint = endpoint.rstrip("/")
                    # Add the metrics path
                    http_endpoint = f"{http_endpoint}/v1/metrics"
                    logger.info(
                        "Adjusted HTTP endpoint from %s to %s (added /v1/metrics path)",
                        endpoint,
                        http_endpoint,
                    )

                exporter = OTLPMetricExporter(
                    endpoint=http_endpoint,
                    timeout=30,  # 30 second timeout for HTTP
                )
                logger.info("Using HTTP/Protobuf exporter for endpoint: %s", http_endpoint)
            except ImportError:
                logger.error(
                    "HTTP/Protobuf exporter not available. "
                    "Install with: pip install opentelemetry-exporter-otlp-proto-http"
                )
                return None
        else:
            # Default to gRPC
            try:
                from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                    OTLPMetricExporter,
                )

                exporter = OTLPMetricExporter(
                    endpoint=endpoint,
                    insecure=insecure,
                )
                logger.info(
                    "Using gRPC exporter for endpoint: %s (insecure=%s)",
                    endpoint,
                    insecure,
                )
            except ImportError:
                logger.error(
                    "gRPC exporter not available. "
                    "Install with: pip install opentelemetry-exporter-otlp-proto-grpc"
                )
                return None

        logger.info(
            "OTLP exporter created: endpoint=%s, protocol=%s, insecure=%s (TLS=%s)",
            endpoint,
            protocol,
            insecure if protocol == "grpc" else "N/A",
            "disabled" if (protocol == "grpc" and insecure) else "enabled",
        )

        # Create metric reader with periodic export
        reader = PeriodicExportingMetricReader(
            exporter=exporter,
            export_interval_millis=export_interval * 1000,
        )

        # Set up meter provider
        provider = MeterProvider(
            resource=resource,
            metric_readers=[reader],
        )
        metrics.set_meter_provider(provider)

        logger.info("OpenTelemetry metrics configured successfully")
        
        # Create a test metric to verify export is working
        test_meter = metrics.get_meter(__name__)
        test_counter = test_meter.create_counter(
            name="memmachine_otel_init",
            description="OTEL metrics initialization counter (test metric)",
        )
        test_counter.add(1, {"status": "initialized"})
        logger.info("Test metric created - will be exported in next interval (~%ds)", export_interval)

        # Return meter for instrumentation
        return test_meter

    except Exception as e:
        logger.exception("Failed to setup OpenTelemetry metrics: %s", e)
        return None


class OTELMetricsFactory(MetricsFactory):
    """
    OpenTelemetry-based implementation of MetricsFactory.

    This bridges MemMachine's metrics interface with OpenTelemetry metrics
    that push to SigNoz OTEL collector.
    """

    def __init__(self, meter: metrics.Meter):
        """Initialize with an OpenTelemetry meter."""
        self._meter = meter
        self._instruments: dict[str, Any] = {}
        logger.debug("OTELMetricsFactory initialized")

    class Counter(MetricsFactory.Counter):
        """OpenTelemetry-based counter implementation."""

        def __init__(self, counter: metrics.Counter):
            """Wrap an OpenTelemetry counter."""
            self._counter = counter

        def increment(
            self,
            value: float = 1,
            labels: dict[str, str] | None = None,
        ) -> None:
            """Increment the counter with optional label values."""
            attributes = labels or {}
            self._counter.add(value, attributes=attributes)

    class Gauge(MetricsFactory.Gauge):
        """OpenTelemetry-based gauge implementation using ObservableGauge."""

        def __init__(self, gauge: metrics.ObservableGauge, name: str):
            """Wrap an OpenTelemetry observable gauge."""
            self._gauge = gauge
            self._name = name
            self._current_value: float = 0.0
            self._current_labels: dict[str, str] = {}

        def set(
            self,
            value: float,
            labels: dict[str, str] | None = None,
        ) -> None:
            """Set the gauge value with optional labels."""
            self._current_value = value
            self._current_labels = labels or {}
            # Note: ObservableGauge uses callbacks, so we store the value
            # The callback will be set up in get_gauge()

    class Histogram(MetricsFactory.Histogram):
        """OpenTelemetry-based histogram implementation."""

        def __init__(self, histogram: metrics.Histogram):
            """Wrap an OpenTelemetry histogram."""
            self._histogram = histogram

        def observe(
            self,
            value: float,
            labels: dict[str, str] | None = None,
        ) -> None:
            """Record a histogram observation with optional labels."""
            attributes = labels or {}
            self._histogram.record(value, attributes=attributes)

    class Summary(MetricsFactory.Summary):
        """
        OpenTelemetry-based summary implementation.

        Note: OpenTelemetry doesn't have a native Summary type,
        so we use Histogram which provides similar functionality.
        """

        def __init__(self, histogram: metrics.Histogram):
            """Wrap an OpenTelemetry histogram as a summary."""
            self._histogram = histogram

        def observe(
            self,
            value: float,
            labels: dict[str, str] | None = None,
        ) -> None:
            """Record a summary observation with optional labels."""
            attributes = labels or {}
            self._histogram.record(value, attributes=attributes)

    def get_counter(
        self,
        name: str,
        description: str,
        label_names: tuple[str, ...] = (),
    ) -> Counter:
        """Return an OpenTelemetry-backed counter, creating it if absent."""
        if name not in self._instruments:
            counter = self._meter.create_counter(
                name=name,
                description=description,
            )
            self._instruments[name] = OTELMetricsFactory.Counter(counter)
            logger.debug("Created OTEL counter: %s", name)

        instrument = self._instruments[name]
        if not isinstance(instrument, OTELMetricsFactory.Counter):
            raise TypeError(f"{name} is not a Counter")

        return instrument

    def get_gauge(
        self,
        name: str,
        description: str,
        label_names: tuple[str, ...] = (),
    ) -> Gauge:
        """Return an OpenTelemetry-backed gauge, creating it if absent."""
        if name not in self._instruments:
            gauge_wrapper = OTELMetricsFactory.Gauge(None, name)  # type: ignore

            def gauge_callback(options):
                """Callback for observable gauge."""
                yield metrics.Observation(
                    gauge_wrapper._current_value, gauge_wrapper._current_labels
                )

            gauge = self._meter.create_observable_gauge(
                name=name,
                callbacks=[gauge_callback],
                description=description,
            )
            gauge_wrapper._gauge = gauge

            self._instruments[name] = gauge_wrapper
            logger.debug("Created OTEL gauge: %s", name)

        instrument = self._instruments[name]
        if not isinstance(instrument, OTELMetricsFactory.Gauge):
            raise TypeError(f"{name} is not a Gauge")

        return instrument

    def get_histogram(
        self,
        name: str,
        description: str,
        label_names: tuple[str, ...] = (),
    ) -> Histogram:
        """Return an OpenTelemetry-backed histogram, creating it if absent."""
        if name not in self._instruments:
            histogram = self._meter.create_histogram(
                name=name,
                description=description,
            )
            self._instruments[name] = OTELMetricsFactory.Histogram(histogram)
            logger.debug("Created OTEL histogram: %s", name)

        instrument = self._instruments[name]
        if not isinstance(instrument, OTELMetricsFactory.Histogram):
            raise TypeError(f"{name} is not a Histogram")

        return instrument

    def get_summary(
        self,
        name: str,
        description: str,
        label_names: tuple[str, ...] = (),
    ) -> Summary:
        """
        Return an OpenTelemetry-backed summary (using histogram), creating it if absent.

        Note: OTEL doesn't have native summaries, but histograms provide
        percentiles and can be used similarly.
        """
        if name not in self._instruments:
            # Use histogram for summary-like behavior
            histogram = self._meter.create_histogram(
                name=name,
                description=description,
            )
            self._instruments[name] = OTELMetricsFactory.Summary(histogram)
            logger.debug("Created OTEL summary (as histogram): %s", name)

        instrument = self._instruments[name]
        if not isinstance(instrument, OTELMetricsFactory.Summary):
            raise TypeError(f"{name} is not a Summary")

        return instrument
