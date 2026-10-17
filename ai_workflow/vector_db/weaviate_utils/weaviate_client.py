import threading
import time
from typing import Optional

import weaviate
from loguru import logger
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.query import Filter
from weaviate.classes.tenants import Tenant

from ai_workflow.exception import VectorStoreError
from ai_workflow.settings import WeaviateConnectionType, config_settings
from ai_workflow.vector_db.sanitization import sanitize_index_name, sanitize_namespace

FILE_NAME_PROPERTY = "file_name"
CONTENT_HASH_PROPERTY = "content_hash"
READY_CACHE_SECONDS = 5.0
TENANT_CACHE_SECONDS = 60.0


class WeaviateConnectionManager:
    """Owns one Weaviate client and reconnects if it drops."""

    def __init__(self) -> None:
        self._client: Optional[weaviate.WeaviateClient] = None
        self._lock = threading.RLock()
        self._health_check = threading.Lock()
        self._ensured: set[str] = set()
        self._ready_checked_at = 0.0
        self._known_tenants: dict[tuple[str, str], float] = {}

    def _connect(self) -> weaviate.WeaviateClient:
        settings = config_settings
        api_key = settings.WEAVIATE_API_KEY.get_secret_value() if settings.WEAVIATE_API_KEY else None
        auth = weaviate.auth.AuthApiKey(api_key) if api_key else None

        if settings.WEAVIATE_CONNECTION_TYPE == WeaviateConnectionType.CLOUD:
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=settings.WEAVIATE_CLUSTER_URL,
                auth_credentials=auth,
            )
        else:
            client = weaviate.connect_to_local(
                host=settings.WEAVIATE_HOST,
                port=settings.WEAVIATE_HTTP_PORT,
                grpc_port=settings.WEAVIATE_GRPC_PORT,
                auth_credentials=auth,
            )
        logger.info(f"Connected to Weaviate ({settings.WEAVIATE_CONNECTION_TYPE.value})")
        return client

    def get_client(self) -> weaviate.WeaviateClient:
        """Return the shared client, health-checked at most every READY_CACHE_SECONDS."""
        client = self._client
        if client is None:
            return self._reconnect(stale=None)
        if time.monotonic() - self._ready_checked_at < READY_CACHE_SECONDS:
            return client

        # Only one thread runs the check; others keep using the current client.
        if not self._health_check.acquire(blocking=False):
            return client
        try:
            healthy = self._probe(client)
        finally:
            self._health_check.release()
        return client if healthy else self._reconnect(stale=client)

    def is_ready(self) -> bool:
        """Uncached readiness check."""
        try:
            return self._probe(self.get_client())
        except Exception:  # noqa: BLE001
            return False

    def _probe(self, client: weaviate.WeaviateClient) -> bool:
        """One health check (call without holding ``_lock``)."""
        try:
            ready = bool(client.is_ready())
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Weaviate health check failed: {error}")
            return False
        if ready:
            self._ready_checked_at = time.monotonic()
        else:
            logger.warning("Weaviate reports not ready")
        return ready

    def _reconnect(self, stale: Optional[weaviate.WeaviateClient]) -> weaviate.WeaviateClient:
        """Replace the stale client (or connect), once across threads."""
        with self._lock:
            if self._client is not None and self._client is not stale:
                return self._client
            self.close()
            try:
                self._client = self._connect()
            except Exception as error:
                raise VectorStoreError(f"Cannot connect to Weaviate: {error}") from error
            self._ready_checked_at = time.monotonic()
            return self._client

    def close(self) -> None:
        with self._lock:
            self._ensured.clear()
            self._known_tenants.clear()
            self._ready_checked_at = 0.0
            if self._client is not None:
                try:
                    self._client.close()
                except Exception as error:  # noqa: BLE001
                    logger.warning(f"Error closing Weaviate client: {error}")
                finally:
                    self._client = None

    def ensure_collection(self, index_name: str) -> str:
        """Create the multi-tenant collection if missing; returns its name."""
        name = sanitize_index_name(index_name)
        if name in self._ensured:
            return name
        with self._lock:
            client = self.get_client()
            if not client.collections.exists(name):
                client.collections.create(
                    name=name,
                    vector_config=Configure.Vectors.self_provided(),
                    multi_tenancy_config=Configure.multi_tenancy(
                        enabled=True,
                        auto_tenant_creation=True,
                        auto_tenant_activation=True,
                    ),
                )
                logger.info(f"Created Weaviate collection {name}")
            self._ensure_properties(client.collections.get(name))
            self._ensured.add(name)
        return name

    @staticmethod
    def _ensure_properties(collection) -> None:
        """Declare the filterable properties up front (Weaviate rejects filters on unknown ones)."""
        existing = {prop.name for prop in collection.config.get().properties}
        for name in (FILE_NAME_PROPERTY, CONTENT_HASH_PROPERTY):
            if name not in existing:
                collection.config.add_property(
                    Property(name=name, data_type=DataType.TEXT, tokenization=Tokenization.FIELD)
                )
                logger.info(f"Added property {name} to {collection.name}")

    def delete_collection(self, index_name: str) -> bool:
        name = sanitize_index_name(index_name)
        self._ensured.discard(name)
        self._forget_collection_tenants(name)
        client = self.get_client()
        if not client.collections.exists(name):
            return False
        client.collections.delete(name)
        return True

    def tenant_exists(self, index_name: str, namespace: str) -> bool:
        """True if the namespace exists (only "yes" is cached)."""
        key = (sanitize_index_name(index_name), sanitize_namespace(namespace))
        seen_at = self._known_tenants.get(key)
        if seen_at is not None and time.monotonic() - seen_at < TENANT_CACHE_SECONDS:
            return True

        name, tenant = key
        client = self.get_client()
        exists = client.collections.exists(name) and client.collections.get(name).tenants.exists(tenant)
        if exists:
            self._known_tenants[key] = time.monotonic()
        return exists

    def forget_tenant(self, index_name: str, namespace: str) -> None:
        """Drop the cached "exists"."""
        self._known_tenants.pop((sanitize_index_name(index_name), sanitize_namespace(namespace)), None)

    def _forget_collection_tenants(self, name: str) -> None:
        with self._lock:
            for key in [key for key in self._known_tenants if key[0] == name]:
                self._known_tenants.pop(key, None)

    def tenant_has_objects(self, index_name: str, namespace: str) -> bool:
        """True if the namespace has at least one chunk."""
        if not self.tenant_exists(index_name, namespace):
            return False
        collection = self.get_client().collections.get(sanitize_index_name(index_name))
        tenant = collection.with_tenant(sanitize_namespace(namespace))
        return (tenant.aggregate.over_all(total_count=True).total_count or 0) > 0

    def clear_tenant(self, index_name: str, namespace: str) -> None:
        """Remove all objects of a tenant but keep the tenant."""
        name = sanitize_index_name(index_name)
        tenant = sanitize_namespace(namespace)
        tenants = self.get_client().collections.get(name).tenants
        tenants.remove(tenant)
        tenants.create(Tenant(name=tenant))

    def delete_tenant(self, index_name: str, namespace: str) -> None:
        self.forget_tenant(index_name, namespace)
        name = sanitize_index_name(index_name)
        self.get_client().collections.get(name).tenants.remove(sanitize_namespace(namespace))

    def get_content_hash(self, index_name: str, namespace: str, file_name: str) -> Optional[str]:
        """Stored content hash for a file, or None."""
        try:
            collection = self.get_client().collections.get(sanitize_index_name(index_name))
            tenant = collection.with_tenant(sanitize_namespace(namespace))
            response = tenant.query.fetch_objects(
                filters=Filter.by_property(FILE_NAME_PROPERTY).equal(file_name),
                return_properties=[CONTENT_HASH_PROPERTY],
                limit=1,
            )
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Could not read the stored content hash for {file_name}: {error}")
            return None
        if not response.objects:
            return None
        return response.objects[0].properties.get(CONTENT_HASH_PROPERTY)

    def delete_by_file_name(self, index_name: str, namespace: str, file_name: str) -> int:
        """Delete a file's chunks in a tenant; returns the count."""
        collection = self.get_client().collections.get(sanitize_index_name(index_name))
        tenant = collection.with_tenant(sanitize_namespace(namespace))
        result = tenant.data.delete_many(
            where=Filter.by_property(FILE_NAME_PROPERTY).equal(file_name)
        )
        return int(result.successful)


db_connection = WeaviateConnectionManager()