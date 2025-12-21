"""Delta Lake dataset integration for evaluation.

Provides unified interface for loading evaluation datasets
from Delta tables and saving results back.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading.

    Args:
        table_path: Path to Delta table or catalog table name.
        id_column: Column containing unique example IDs.
        input_column: Column containing input text/prompts.
        reference_column: Column containing reference/ground truth.
        metadata_columns: Additional columns to preserve.
        filter_condition: SQL filter to apply when loading.
        sample_size: Number of examples to sample (None for all).
        sample_seed: Random seed for sampling.
    """

    table_path: str
    id_column: str = "id"
    input_column: str = "input"
    reference_column: str = "reference"
    metadata_columns: list[str] = field(default_factory=list)
    filter_condition: str | None = None
    sample_size: int | None = None
    sample_seed: int = 42


class DeltaDataset:
    """Wrapper for Delta Lake evaluation datasets.

    Provides methods for loading, sampling, and saving evaluation data
    with full Delta Lake features (versioning, time travel, etc.).
    """

    def __init__(
        self,
        spark: SparkSession,
        config: DatasetConfig,
    ):
        """Initialize dataset.

        Args:
            spark: Active SparkSession.
            config: Dataset configuration.
        """
        self.spark = spark
        self.config = config
        self._df: DataFrame | None = None
        self._count: int | None = None

    def load(self) -> DataFrame:
        """Load dataset from Delta table.

        Returns:
            Spark DataFrame with dataset.
        """
        logger.info(f"Loading dataset from {self.config.table_path}")

        # try Delta format first, fall back to parquet
        try:
            df = self.spark.read.format("delta").load(self.config.table_path)
        except Exception:
            # might be a catalog table or parquet
            try:
                df = self.spark.table(self.config.table_path)
            except Exception:
                df = self.spark.read.parquet(self.config.table_path)

        # apply filter if specified
        if self.config.filter_condition:
            df = df.filter(self.config.filter_condition)

        # select required columns
        columns = [
            self.config.id_column,
            self.config.input_column,
            self.config.reference_column,
        ] + self.config.metadata_columns

        # only select columns that exist
        existing_cols = [c for c in columns if c in df.columns]
        df = df.select(*existing_cols)

        # sample if requested
        if self.config.sample_size is not None:
            total = df.count()
            if total > self.config.sample_size:
                fraction = self.config.sample_size / total
                df = df.sample(
                    withReplacement=False,
                    fraction=min(fraction * 1.2, 1.0),  # oversample slightly
                    seed=self.config.sample_seed,
                ).limit(self.config.sample_size)
            logger.info(f"Sampled {self.config.sample_size} from {total} examples")

        self._df = df
        return df

    @property
    def dataframe(self) -> DataFrame:
        """Get loaded dataframe, loading if necessary."""
        if self._df is None:
            self.load()
        return self._df

    def count(self) -> int:
        """Get number of examples in dataset."""
        if self._count is None:
            self._count = self.dataframe.count()
        return self._count

    def get_version_info(self) -> dict[str, Any] | None:
        """Get Delta table version information if available."""
        try:
            from delta.tables import DeltaTable

            dt = DeltaTable.forPath(self.spark, self.config.table_path)
            history = dt.history(1).collect()
            if history:
                row = history[0]
                return {
                    "version": row.version,
                    "timestamp": str(row.timestamp),
                    "operation": row.operation,
                }
        except Exception as e:
            logger.debug(f"Could not get Delta version info: {e}")
        return None


def load_dataset(
    spark: SparkSession,
    table_path: str,
    id_column: str = "id",
    input_column: str = "input",
    reference_column: str = "reference",
    filter_condition: str | None = None,
    sample_size: int | None = None,
) -> DataFrame:
    """Convenience function to load a dataset.

    Args:
        spark: Active SparkSession.
        table_path: Path to Delta table or table name.
        id_column: Column for example IDs.
        input_column: Column for input text.
        reference_column: Column for reference text.
        filter_condition: Optional SQL filter.
        sample_size: Optional sample size.

    Returns:
        Loaded DataFrame.

    Example:
        df = load_dataset(
            spark,
            "catalog.schema.eval_data",
            sample_size=1000,
        )
    """
    config = DatasetConfig(
        table_path=table_path,
        id_column=id_column,
        input_column=input_column,
        reference_column=reference_column,
        filter_condition=filter_condition,
        sample_size=sample_size,
    )
    dataset = DeltaDataset(spark, config)
    return dataset.load()


def save_results(
    df: DataFrame,
    output_path: str,
    mode: str = "overwrite",
    partition_by: list[str] | None = None,
) -> None:
    """Save evaluation results to Delta table.

    Args:
        df: DataFrame with results.
        output_path: Path to save results.
        mode: Save mode ("overwrite", "append", "error").
        partition_by: Columns to partition by.

    Example:
        save_results(results_df, "catalog.schema.eval_results")
    """
    logger.info(f"Saving results to {output_path}")

    writer = df.write.format("delta").mode(mode)

    if partition_by:
        writer = writer.partitionBy(*partition_by)

    writer.save(output_path)
    logger.info(f"Saved {df.count()} results to {output_path}")


def create_eval_dataset(
    spark: SparkSession,
    data: list[dict[str, Any]],
    id_column: str = "id",
    input_column: str = "input",
    reference_column: str = "reference",
) -> DataFrame:
    """Create evaluation dataset from list of dictionaries.

    Useful for testing or small-scale evaluations.

    Args:
        spark: Active SparkSession.
        data: List of dicts with at least id, input, and reference keys.
        id_column: Name of ID column.
        input_column: Name of input column.
        reference_column: Name of reference column.

    Returns:
        Spark DataFrame with evaluation data.

    Example:
        data = [
            {"id": "1", "input": "What is 2+2?", "reference": "4"},
            {"id": "2", "input": "Capital of France?", "reference": "Paris"},
        ]
        df = create_eval_dataset(spark, data)
    """
    if not data:
        raise ValueError("Data cannot be empty")

    # validate required columns present
    required = {id_column, input_column, reference_column}
    first_keys = set(data[0].keys())
    missing = required - first_keys
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return spark.createDataFrame(data)
