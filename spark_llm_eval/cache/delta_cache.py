"""Delta Lake-backed response cache manager."""

import json
import logging
from datetime import datetime, timedelta

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    MapType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from spark_llm_eval.cache.config import CacheConfig, CachePolicy
from spark_llm_eval.cache.key_generator import generate_cache_key, generate_prompt_hash
from spark_llm_eval.cache.stats import CacheStatistics
from spark_llm_eval.core.config import ModelConfig

logger = logging.getLogger(__name__)


# Delta table schema for cache
CACHE_TABLE_SCHEMA = StructType(
    [
        # Primary key - content-addressable hash
        StructField("cache_key", StringType(), nullable=False),
        # Request metadata
        StructField("prompt_hash", StringType(), nullable=False),
        StructField("model_name", StringType(), nullable=False),
        StructField("provider", StringType(), nullable=False),
        StructField("temperature", FloatType(), nullable=False),
        StructField("max_tokens", IntegerType(), nullable=False),
        # Response data
        StructField("response_text", StringType(), nullable=True),
        StructField("input_tokens", IntegerType(), nullable=True),
        StructField("output_tokens", IntegerType(), nullable=True),
        StructField("finish_reason", StringType(), nullable=True),
        StructField("error", StringType(), nullable=True),
        # Cost tracking
        StructField("cost_usd", FloatType(), nullable=True),
        StructField("latency_ms", FloatType(), nullable=True),
        # Versioning and TTL
        StructField("created_at", TimestampType(), nullable=False),
        StructField("last_accessed_at", TimestampType(), nullable=True),
        StructField("access_count", IntegerType(), nullable=False),
        StructField("cache_version", StringType(), nullable=False),
        # Provenance
        StructField("task_id", StringType(), nullable=True),
        StructField("run_id", StringType(), nullable=True),
        # Extensibility
        StructField("metadata", MapType(StringType(), StringType()), nullable=True),
    ]
)


def create_cache_table(spark: SparkSession, table_path: str) -> None:
    """Create the cache Delta table with optimal settings.

    Args:
        spark: Active SparkSession
        table_path: Path for the Delta table
    """
    # Create empty DataFrame with schema
    empty_df = spark.createDataFrame([], CACHE_TABLE_SCHEMA)

    # Write as Delta table
    # Note: autoOptimize options are Databricks-specific and not used here
    # for compatibility with open-source Delta Lake
    (
        empty_df.write.format("delta")
        .mode("ignore")  # Don't overwrite if exists
        .save(table_path)
    )

    logger.info(f"Created cache table at {table_path}")


class CacheError(Exception):
    """Exception raised for cache-related errors."""

    pass


class DeltaCacheManager:
    """Manages Delta Lake-backed response cache.

    Provides content-addressable caching of LLM responses using Delta Lake
    for storage. Supports partial cache hits, TTL-based expiry, version-based
    invalidation, and detailed statistics tracking.

    Example:
        >>> from spark_llm_eval.cache import CacheConfig, CachePolicy, DeltaCacheManager
        >>> config = CacheConfig(
        ...     policy=CachePolicy.ENABLED,
        ...     table_path="/tmp/cache",
        ...     ttl_hours=24,
        ... )
        >>> manager = DeltaCacheManager(spark, config, model_config)
        >>> data_with_keys = manager.add_cache_keys(data)
        >>> cached_df, uncached_df = manager.lookup(data_with_keys)
    """

    def __init__(
        self,
        spark: SparkSession,
        config: CacheConfig,
        model_config: ModelConfig,
    ):
        """Initialize cache manager.

        Args:
            spark: Active SparkSession
            config: Cache configuration
            model_config: Model configuration (for key generation)
        """
        self.spark = spark
        self.config = config
        self.model_config = model_config
        self._stats = CacheStatistics()
        self._ensure_table_exists()

    def _ensure_table_exists(self) -> None:
        """Create cache table if it doesn't exist."""
        if self.config.policy == CachePolicy.DISABLED:
            return

        try:
            from delta.tables import DeltaTable

            if not DeltaTable.isDeltaTable(self.spark, self.config.table_path):
                logger.info(f"Creating cache table at {self.config.table_path}")
                create_cache_table(self.spark, self.config.table_path)
        except Exception as e:
            logger.warning(f"Could not verify/create cache table: {e}")
            # Table might exist but DeltaTable.isDeltaTable failed
            # We'll handle errors during actual operations

    def add_cache_keys(self, df: DataFrame, prompt_column: str = "prompt") -> DataFrame:
        """Add cache_key column to DataFrame.

        Generates SHA-256 cache keys based on prompt and model configuration.

        Args:
            df: DataFrame with prompt column
            prompt_column: Name of the prompt column

        Returns:
            DataFrame with added 'cache_key' column
        """
        # Prepare model config values for UDF closure
        model_name = self.model_config.model_name
        provider = self.model_config.provider.value
        temperature = float(self.model_config.temperature)
        max_tokens = int(self.model_config.max_tokens)
        extra_params = self.model_config.extra_params or {}
        extra_params_json = json.dumps(extra_params, sort_keys=True)

        @F.udf(StringType())
        def compute_cache_key(prompt: str) -> str:
            if prompt is None:
                return None
            extra = json.loads(extra_params_json) if extra_params_json != "{}" else None
            return generate_cache_key(
                prompt=prompt,
                model_name=model_name,
                provider=provider,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_params=extra,
            )

        return df.withColumn("cache_key", compute_cache_key(F.col(prompt_column)))

    def lookup(self, df: DataFrame) -> tuple[DataFrame, DataFrame]:
        """Look up cached responses.

        Performs a left join with the cache table to find cached responses.
        Filters by model, provider, cache version, and TTL.

        Args:
            df: DataFrame with 'cache_key' column

        Returns:
            Tuple of (cached_df, uncached_df):
            - cached_df: Rows with cache hits, including response data
            - uncached_df: Rows with cache misses, need inference
        """
        if self.config.policy in (CachePolicy.DISABLED, CachePolicy.WRITE_ONLY):
            # DISABLED: No caching at all
            # WRITE_ONLY: Store only, don't look up
            return self.spark.createDataFrame([], df.schema), df

        # Read cache table with TTL filter
        cache_df = self._read_cache_with_ttl()

        # Filter by current model/provider for partition pruning
        cache_df = cache_df.filter(
            (F.col("model_name") == self.model_config.model_name)
            & (F.col("provider") == self.model_config.provider.value)
            & (F.col("cache_version") == self.config.cache_version)
        )

        # Select only needed columns from cache
        cache_cols = cache_df.select(
            F.col("cache_key").alias("_cache_key"),
            F.col("response_text").alias("_cached_response"),
            F.col("input_tokens").alias("_cached_input_tokens"),
            F.col("output_tokens").alias("_cached_output_tokens"),
            F.col("cost_usd").alias("_cached_cost"),
            F.col("latency_ms").alias("_cached_latency"),
        )

        # Left join to find matches
        joined = df.join(cache_cols, df.cache_key == cache_cols._cache_key, how="left")

        # Split into cached and uncached
        cached_df = (
            joined.filter(F.col("_cached_response").isNotNull())
            .withColumn("prediction", F.col("_cached_response"))
            .withColumn("input_tokens", F.col("_cached_input_tokens"))
            .withColumn("output_tokens", F.col("_cached_output_tokens"))
            .withColumn("cost_usd", F.col("_cached_cost"))
            .withColumn("latency_ms", F.col("_cached_latency"))
            .withColumn("from_cache", F.lit(True))
            .drop(
                "_cache_key",
                "_cached_response",
                "_cached_input_tokens",
                "_cached_output_tokens",
                "_cached_cost",
                "_cached_latency",
            )
        )

        uncached_df = joined.filter(F.col("_cached_response").isNull()).drop(
            "_cache_key",
            "_cached_response",
            "_cached_input_tokens",
            "_cached_output_tokens",
            "_cached_cost",
            "_cached_latency",
        )

        # Update statistics
        cached_count = cached_df.count()
        uncached_count = uncached_df.count()
        self._stats.record_lookups(cached_count, uncached_count)

        logger.info(f"Cache lookup: {cached_count} hits, {uncached_count} misses")

        return cached_df, uncached_df

    def store(
        self,
        df: DataFrame,
        task_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Store inference results in cache.

        Uses Delta MERGE for upsert semantics - updates access time for existing
        entries, inserts new entries.

        Args:
            df: DataFrame with cache_key, prompt, response columns
            task_id: Optional task ID for provenance
            run_id: Optional MLflow run ID for provenance
        """
        if self.config.policy in (CachePolicy.DISABLED, CachePolicy.READ_ONLY):
            return

        from delta.tables import DeltaTable

        now = datetime.utcnow()

        # Prepare data for cache - create prompt_hash from prompt column
        @F.udf(StringType())
        def compute_prompt_hash(prompt: str) -> str:
            if prompt is None:
                return None
            return generate_prompt_hash(prompt)

        # Build cache data with all required columns
        cache_data = (
            df.filter(F.col("response_text").isNotNull())  # Don't cache errors
            .withColumn("prompt_hash", compute_prompt_hash(F.col("prompt")))
            .withColumn("model_name", F.lit(self.model_config.model_name))
            .withColumn("provider", F.lit(self.model_config.provider.value))
            .withColumn("temperature", F.lit(float(self.model_config.temperature)))
            .withColumn("max_tokens", F.lit(int(self.model_config.max_tokens)))
            .withColumn("created_at", F.lit(now))
            .withColumn("last_accessed_at", F.lit(now))
            .withColumn("access_count", F.lit(1))
            .withColumn("cache_version", F.lit(self.config.cache_version))
            .withColumn("task_id", F.lit(task_id))
            .withColumn("run_id", F.lit(run_id))
            .withColumn("metadata", F.lit(None).cast("map<string,string>"))
        )

        # Select columns matching schema
        cache_data = cache_data.select(
            "cache_key",
            "prompt_hash",
            "model_name",
            "provider",
            "temperature",
            "max_tokens",
            "response_text",
            F.col("input_tokens").cast("int"),
            F.col("output_tokens").cast("int"),
            F.col("finish_reason").cast("string")
            if "finish_reason" in df.columns
            else F.lit(None).cast("string").alias("finish_reason"),
            F.col("error").cast("string")
            if "error" in df.columns
            else F.lit(None).cast("string").alias("error"),
            F.col("cost_usd").cast("float"),
            F.col("latency_ms").cast("float"),
            "created_at",
            "last_accessed_at",
            "access_count",
            "cache_version",
            "task_id",
            "run_id",
            "metadata",
        )

        try:
            # Count before merge (DataFrame may be invalidated after merge)
            stored_count = cache_data.count()
            if stored_count == 0:
                return

            # MERGE into cache table (upsert)
            cache_table = DeltaTable.forPath(self.spark, self.config.table_path)

            (
                cache_table.alias("cache")
                .merge(cache_data.alias("new"), "cache.cache_key = new.cache_key")
                .whenMatchedUpdate(
                    set={
                        "last_accessed_at": F.col("new.last_accessed_at"),
                        "access_count": F.col("cache.access_count") + 1,
                    }
                )
                .whenNotMatchedInsertAll()
                .execute()
            )

            self._stats.record_stores(stored_count)
            logger.info(f"Stored {stored_count} responses in cache")

        except Exception as e:
            logger.error(f"Failed to store in cache: {e}")
            # Don't fail the evaluation if cache write fails

    def _read_cache_with_ttl(self) -> DataFrame:
        """Read cache table, filtering out expired entries.

        Returns:
            DataFrame with non-expired cache entries
        """
        try:
            cache_df = self.spark.read.format("delta").load(self.config.table_path)

            if self.config.ttl_hours:
                cutoff = datetime.utcnow() - timedelta(hours=self.config.ttl_hours)
                cache_df = cache_df.filter(F.col("created_at") >= cutoff)

            return cache_df

        except Exception as e:
            logger.warning(f"Could not read cache table: {e}")
            # Return empty DataFrame with schema
            return self.spark.createDataFrame([], CACHE_TABLE_SCHEMA)

    def invalidate(
        self,
        model_name: str | None = None,
        before_timestamp: datetime | None = None,
        cache_version: str | None = None,
    ) -> int:
        """Invalidate (delete) cache entries matching criteria.

        Args:
            model_name: Invalidate entries for this model only
            before_timestamp: Invalidate entries created before this time
            cache_version: Invalidate entries with this version

        Returns:
            Number of entries invalidated

        Raises:
            ValueError: If no invalidation criterion is provided
        """
        from delta.tables import DeltaTable

        conditions = []
        if model_name:
            conditions.append(f"model_name = '{model_name}'")
        if before_timestamp:
            conditions.append(f"created_at < '{before_timestamp.isoformat()}'")
        if cache_version:
            conditions.append(f"cache_version = '{cache_version}'")

        if not conditions:
            raise ValueError("At least one invalidation criterion required")

        condition = " AND ".join(conditions)

        # Count before delete
        count_before = (
            self.spark.read.format("delta").load(self.config.table_path).filter(condition).count()
        )

        cache_table = DeltaTable.forPath(self.spark, self.config.table_path)
        cache_table.delete(condition)

        logger.info(f"Invalidated {count_before} cache entries")
        return count_before

    def get_statistics(self) -> CacheStatistics:
        """Get cache statistics for this session.

        Returns:
            CacheStatistics object with hit/miss counts and cost savings
        """
        return self._stats

    def vacuum(self, retention_hours: int = 168) -> None:
        """Run VACUUM on cache table to clean up old files.

        Args:
            retention_hours: Minimum file retention in hours (default 7 days)
        """
        from delta.tables import DeltaTable

        cache_table = DeltaTable.forPath(self.spark, self.config.table_path)
        cache_table.vacuum(retention_hours / 24)  # vacuum takes days
        logger.info(f"Vacuumed cache table with {retention_hours}h retention")

    def get_cache_info(self) -> dict:
        """Get information about the cache table.

        Returns:
            Dictionary with cache table metadata
        """
        try:
            cache_df = self.spark.read.format("delta").load(self.config.table_path)
            total_entries = cache_df.count()

            # Get entry counts by model
            by_model = cache_df.groupBy("model_name", "provider").count().collect()

            return {
                "table_path": self.config.table_path,
                "total_entries": total_entries,
                "cache_version": self.config.cache_version,
                "ttl_hours": self.config.ttl_hours,
                "entries_by_model": [
                    {"model": r.model_name, "provider": r.provider, "count": r["count"]}
                    for r in by_model
                ],
            }
        except Exception as e:
            return {"error": str(e)}
