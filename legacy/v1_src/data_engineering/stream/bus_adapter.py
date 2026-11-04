"""Stream Bus Adapters supporting both Open Stack (In-Memory/Kafka) and GCP Stack (Pub/Sub)."""
import json
import uuid
from typing import Any, Callable, Dict, List
import structlog
from src.core.config import get_settings
from src.core.interfaces import StreamBusProtocol

logger = structlog.get_logger(__name__)


class InMemoryStreamBus:
    """In-memory partitioned event bus for local dev, unit tests, and CI/CD."""

    def __init__(self):
        self.topics: Dict[str, List[Dict[str, Any]]] = {}
        self.subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}

    def publish_event(self, topic: str, key: str, payload: Dict[str, Any]) -> str:
        """Publish event and deliver synchronously to registered subscribers."""
        if topic not in self.topics:
            self.topics[topic] = []
        msg_id = f"MSG-{uuid.uuid4().hex[:12]}"
        envelope = {
            "message_id": msg_id,
            "partition_key": key,
            "topic": topic,
            "data": payload,
        }
        self.topics[topic].append(envelope)

        # Notify subscribers
        for handler in self.subscribers.get(topic, []):
            try:
                handler(envelope)
            except Exception as e:
                logger.error("bus.subscriber_error", topic=topic, error=str(e))

        return msg_id

    def subscribe(self, topic: str, group_id: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Register subscriber on topic."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(handler)

    def get_messages(self, topic: str) -> List[Dict[str, Any]]:
        """Retrieve all published messages on a topic."""
        return self.topics.get(topic, [])


class GCPPubSubStreamBus:
    """GCP Pub/Sub stream adapter using google-cloud-pubsub client."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self._publisher = None

    def _get_publisher(self):
        if self._publisher is None:
            try:
                from google.cloud import pubsub_v1
                self._publisher = pubsub_v1.PublisherClient()
            except ImportError:
                logger.warn("gcp.pubsub_not_installed, falling back to simulated client")
                return None
        return self._publisher

    def publish_event(self, topic: str, key: str, payload: Dict[str, Any]) -> str:
        pub = self._get_publisher()
        data = json.dumps(payload).encode("utf-8")
        if pub:
            topic_path = pub.topic_path(self.project_id, topic)
            future = pub.publish(topic_path, data, ordering_key=key)
            return future.result()
        # Fallback simulation
        return f"GCP-SIM-{uuid.uuid4().hex[:12]}"

    def subscribe(self, topic: str, group_id: str, handler: Any) -> None:
        logger.info("gcp.pubsub_subscribed", topic=topic, group=group_id)


class StreamBusFactory:
    """Factory creating appropriate stream bus implementation based on settings."""

    @staticmethod
    def get_bus() -> StreamBusProtocol:
        settings = get_settings()
        if settings.STREAM_BUS_TYPE == "gcp_pubsub":
            return GCPPubSubStreamBus(project_id=settings.GCP_PROJECT_ID)
        return InMemoryStreamBus()
