import argparse

import numpy as np


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate workflow for fitting simulation/markdown model for cannibalization optimization."
    )
    parser.add_argument("--stage", type=str, default="dev")
    parser.add_argument("--artifact_source", type=str, required=True)
    parser.add_argument("--artifact_path", type=str, required=True)
    parser.add_argument("--variables_file", type=str, required=True)
    return parser.parse_args()


def output_variables(variable_dict: dict[str, str]):
    with open("variables.txt", "w") as f:
        for key, value in variable_dict.items():
            f.write(f"{key}={value}\n")


def output_report():
    report = """
# 機械学習モデル検証レポート

## プロジェクト概要
- **プロジェクト名:** 
- **目的:** 
- **モデル:** 
- **データセット:** 

## 実行環境
- **OS:** 
- **Pythonバージョン:** 
- **ライブラリとバージョン:**
  - pandas: 
  - numpy: 
  - scikit-learn: 
  - PyTorch: 
  - その他:

## データ準備
### データセットの説明
- **データセット名:** 
- **サイズ:** 
- **特徴量:** 
  - 特徴量1: 説明
  - 特徴量2: 説明
  - ...

### データ前処理
```python
# サンプルコード: データ前処理
import pandas as pd

# データの読み込み
df = pd.read_csv('path_to_dataset.csv')

# データクリーニング
df = df.dropna()

# 特徴量エンジニアリング
df['new_feature'] = df['existing_feature'] * 2
    """

    with open("report.md", "w") as f:
        f.write(report)


def main(args):
    print(args.stage)
    print(args.artifact_source)
    print(args.artifact_path)
    print(args.variables_file)

    print("numpy check", np.__version__)

    variables = {
        "TIMESTAMP": "2021-09-01-17-00-00",
        "LOCAL_ARTIFACT_PATH": args.artifact_path,
        "LOCAL_ARTIFACT_META_DATA_PATH": args.artifact_source,
    }
    output_variables(variables)

    output_report()


if __name__ == "__main__":
    args = _parse()
    main(args)
