from typing import Dict

from pakkr import Pipeline, returns
from pandas import DataFrame
from sentiment_analysis.data_loaders.huggingface_loader import HuggingfaceLoader


@returns(DataFrame)
def load_data(dataset_name: str) -> DataFrame:
    loader = HuggingfaceLoader(dataset_name)
    return loader.load_test_data()


@returns(Dict)
def generate_reports(data) -> Dict:
    uniques = data.label.unique()
    data["num_chars"] = data.apply(lambda row: len(row.text), axis=1)

    report = {
        "Num of examples": len(data),
        "Unique labels": str(uniques),
        "Num of unique labels": len(uniques),
        "Min text length": data.num_chars.min(),
        "Max text length": data.num_chars.max(),
        "Mean text length": data.num_chars.mean(),
        "Example": {"Text": data.iloc[0].text, "Label": data.iloc[0].label},
    }

    # TODO: do not use print
    print(report)
    return report


pipeline = Pipeline(load_data, generate_reports, _name="data inspection")
