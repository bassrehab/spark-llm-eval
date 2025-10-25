"""Tests for datasets module."""

import pytest
from unittest.mock import MagicMock, patch
from spark_llm_eval.datasets.delta_dataset import (
    DatasetConfig,
    DeltaDataset,
    create_eval_dataset,
)


class TestDatasetConfig:
    """Tests for DatasetConfig."""

    def test_default_values(self):
        config = DatasetConfig(table_path="test/path")
        assert config.table_path == "test/path"
        assert config.id_column == "id"
        assert config.input_column == "input"
        assert config.reference_column == "reference"
        assert config.metadata_columns == []
        assert config.filter_condition is None
        assert config.sample_size is None

    def test_custom_values(self):
        config = DatasetConfig(
            table_path="catalog.schema.table",
            id_column="example_id",
            input_column="question",
            reference_column="answer",
            metadata_columns=["category", "difficulty"],
            filter_condition="category = 'math'",
            sample_size=100,
        )
        assert config.table_path == "catalog.schema.table"
        assert config.id_column == "example_id"
        assert config.sample_size == 100
        assert len(config.metadata_columns) == 2


class TestDeltaDataset:
    """Tests for DeltaDataset."""

    def test_initialization(self):
        mock_spark = MagicMock()
        config = DatasetConfig(table_path="test/path")
        dataset = DeltaDataset(mock_spark, config)

        assert dataset.spark == mock_spark
        assert dataset.config == config
        assert dataset._df is None

    @patch("spark_llm_eval.datasets.delta_dataset.DeltaDataset.load")
    def test_dataframe_property_triggers_load(self, mock_load):
        mock_spark = MagicMock()
        mock_df = MagicMock()
        mock_load.return_value = mock_df

        config = DatasetConfig(table_path="test/path")
        dataset = DeltaDataset(mock_spark, config)
        dataset._df = None

        # access property should trigger load
        _ = dataset.dataframe
        # first access loads
        # subsequent access should use cached
        dataset._df = mock_df
        _ = dataset.dataframe
        # load called only once since we set _df


class TestCreateEvalDataset:
    """Tests for create_eval_dataset function."""

    def test_empty_data_raises(self):
        mock_spark = MagicMock()
        with pytest.raises(ValueError, match="Data cannot be empty"):
            create_eval_dataset(mock_spark, [])

    def test_missing_required_columns_raises(self):
        mock_spark = MagicMock()
        data = [{"id": "1", "input": "test"}]  # missing reference

        with pytest.raises(ValueError, match="Missing required columns"):
            create_eval_dataset(mock_spark, data)

    def test_valid_data(self):
        mock_spark = MagicMock()
        mock_df = MagicMock()
        mock_spark.createDataFrame.return_value = mock_df

        data = [
            {"id": "1", "input": "What is 2+2?", "reference": "4"},
            {"id": "2", "input": "Capital of France?", "reference": "Paris"},
        ]

        result = create_eval_dataset(mock_spark, data)

        mock_spark.createDataFrame.assert_called_once_with(data)
        assert result == mock_df

    def test_custom_column_names(self):
        mock_spark = MagicMock()
        mock_df = MagicMock()
        mock_spark.createDataFrame.return_value = mock_df

        data = [{"example_id": "1", "question": "test", "answer": "result"}]

        result = create_eval_dataset(
            mock_spark,
            data,
            id_column="example_id",
            input_column="question",
            reference_column="answer",
        )

        mock_spark.createDataFrame.assert_called_once()
