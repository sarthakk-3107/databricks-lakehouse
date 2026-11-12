"""Shared, testable code for the Financial Transaction Lakehouse + Fraud ML project.

Notebooks import from here so that logic (data generation, feature definitions,
metrics, cost-based thresholding) lives in one place and can be unit-tested
outside Databricks.
"""

__version__ = "2.0.0"
