import pandas as pd
from IPython.display import Markdown, display


def _describe_datatypes(dataframe: pd.DataFrame) -> None:
    display(Markdown('### Data Types'))
    display(dataframe.dtypes)


def _describe_missing_values(dataframe: pd.DataFrame) -> None:
    display(Markdown('### Missing Values'))
    missing_values = dataframe.isnull().sum()
    columns_with_missing_values = missing_values[missing_values > 0]
    display(columns_with_missing_values)


def _describe_size(dataframe: pd.DataFrame) -> None:
    display(Markdown('### Dataframe Size'))
    display(Markdown(f'Number of rows: {dataframe.shape[0]}'))
    display(Markdown(f'Number of columns: {dataframe.shape[1]}'))


def _describe_head(dataframe: pd.DataFrame, n: int = 5) -> None:
    display(Markdown(f'### First {n} Rows of the Dataframe'))
    display(dataframe.head(n))


def _describe_summary_statistics(dataframe: pd.DataFrame) -> None:
    display(Markdown('### Statistical Summary'))
    display(Markdown('#### Continuous Variables'))
    display(dataframe.describe(include=['number']).round(2).transpose())
    display(Markdown('#### Categorical Variables'))
    display(dataframe.describe(include=['category']).transpose())


def describe_univariate(dataframe: pd.DataFrame) -> None:
    display(Markdown('## Descriptive Analysis'))
    _describe_datatypes(dataframe)
    _describe_missing_values(dataframe)
    _describe_size(dataframe)
    _describe_head(dataframe)
    _describe_summary_statistics(dataframe)
