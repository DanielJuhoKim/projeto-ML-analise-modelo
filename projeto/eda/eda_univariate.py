import pandas as pd
from IPython.display import Markdown, display

from ._descriptive_univariate import describe_univariate
from ._visualization_univariate import visualize_univariate


def run_univariate_EDA(dataframe: pd.DataFrame) -> None:
    display(Markdown('# Univariate EDA\n---\n'))
    describe_univariate(dataframe)
    display(Markdown('---\n'))
    visualize_univariate(dataframe)
