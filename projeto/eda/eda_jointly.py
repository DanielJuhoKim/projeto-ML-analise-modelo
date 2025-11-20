import pandas as pd
from IPython.display import Markdown, display

from ._descriptive_jointly import describe_associations
from ._visualization_jointly import visualize_jointly


def run_joint_EDA(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    display(Markdown('# Bivariate EDA\n---\n'))
    describe_associations(X_train, y_train)
    visualize_jointly(X_train, y_train)
