"""Central naming for catalogs, schemas, tables and volume paths.

Default catalog is `workspace`, which exists in Databricks Free Edition.
Everything else lives in schemas prefixed with `fin_` so the project never
collides with other work in the same catalog.
"""
from __future__ import annotations

from dataclasses import dataclass

LAYERS = ("raw", "bronze", "silver", "gold", "ml")


@dataclass(frozen=True)
class Paths:
    catalog: str = "workspace"
    prefix: str = "fin"
    volume: str = "files"

    # ---- Unity Catalog objects -------------------------------------------
    def schema(self, layer: str) -> str:
        if layer not in LAYERS:
            raise ValueError(f"unknown layer {layer!r}; expected one of {LAYERS}")
        return f"{self.catalog}.{self.prefix}_{layer}"

    def t(self, layer: str, name: str) -> str:
        """Fully qualified table name, e.g. P.t('silver', 'transactions')."""
        return f"{self.schema(layer)}.{name}"

    # ---- Volume paths (files) --------------------------------------------
    @property
    def volume_root(self) -> str:
        return f"/Volumes/{self.catalog}/{self.prefix}_raw/{self.volume}"

    @property
    def landing(self) -> str:
        """Where raw files land; Auto Loader watches this."""
        return f"{self.volume_root}/landing"

    @property
    def pending(self) -> str:
        """Held-back files you can move into `landing` to demo incremental loads."""
        return f"{self.volume_root}/pending"

    @property
    def checkpoints(self) -> str:
        return f"{self.volume_root}/checkpoints"

    @property
    def public_data(self) -> str:
        """Drop public datasets (e.g. Sparkov CSVs) here before adapting them."""
        return f"{self.volume_root}/public"
