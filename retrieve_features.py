from pathlib import Path

import pandas as pd
from feast import FeatureStore


ROOT = Path(__file__).resolve().parent


def main():
    store = FeatureStore(repo_path=str(ROOT / "feature_repo"))
    entity_df = pd.DataFrame(
        {
            "digit_id": [0],
            "event_timestamp": [pd.Timestamp.now(tz="UTC")],
        }
    )
    features = store.get_historical_features(
        entity_df=entity_df,
        features=[f"digit_features:pixel_{index}" for index in range(4)],
    ).to_df()
    print(features)


if __name__ == "__main__":
    main()
