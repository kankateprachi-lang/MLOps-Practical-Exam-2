from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from feast.value_type import ValueType


digit = Entity(
    name="digit_id",
    join_keys=["digit_id"],
    value_type=ValueType.INT64,
)

digit_source = FileSource(
    name="digit_source",
    path="data/digit_features.parquet",
    timestamp_field="event_timestamp",
)

pixel_fields = [
    Field(name=f"pixel_{index}", dtype=Float32) for index in range(64)
]

digit_features = FeatureView(
    name="digit_features",
    entities=[digit],
    ttl=timedelta(days=3650),
    schema=pixel_fields + [Field(name="target", dtype=Int64)],
    source=digit_source,
)
