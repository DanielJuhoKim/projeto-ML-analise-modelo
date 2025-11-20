import pandas as pd
from IPython.display import Markdown, display
from scipy.stats import contingency


def _compute_correlation(df_numeric: pd.DataFrame) -> pd.DataFrame:
    correlation_matrix = df_numeric.corr(method='pearson')
    return correlation_matrix


def _compute_cramer_v(df_categorical: pd.DataFrame) -> pd.DataFrame:
    association_data = []

    cols = df_categorical.columns
    for col_1 in cols:
        for col_2 in cols:
            # Trivial case.
            if col_1 == col_2:
                association_data.append((col_1, col_2, 1.0))
                continue

            data_1 = df_categorical[col_1]
            data_2 = df_categorical[col_2]

            contingency_table = pd.crosstab(data_1, data_2)
            cramer_v = contingency.association(contingency_table, method='cramer')
            association_data.append((col_1, col_2, cramer_v))

    association_matrix = pd.DataFrame.from_records(
        association_data,
        columns=[
            'Variable 1',
            'Variable 2',
            'Cramer V',
        ],
    ).pivot(index='Variable 1', columns='Variable 2', values='Cramer V')

    return association_matrix


def _compute_categorical_numerical_association(
    df_categorical: pd.DataFrame,
    df_numerical: pd.DataFrame,
) -> pd.DataFrame:
    association_data = []

    categorical_cols = df_categorical.columns
    numerical_cols = df_numerical.columns

    for cat_col in categorical_cols:
        for num_col in numerical_cols:
            # Discretize the numerical feature into quantiles.
            discretized_num = pd.qcut(df_numerical[num_col], q=4, duplicates='drop')

            contingency_table = pd.crosstab(df_categorical[cat_col], discretized_num)
            cramer_v = contingency.association(contingency_table, method='cramer')
            association_data.append((cat_col, num_col, cramer_v))

    association_matrix = pd.DataFrame.from_records(
        association_data,
        columns=[
            'Categorical Variable',
            'Numerical Variable',
            'Cramer V',
        ],
    ).pivot(
        index='Categorical Variable', columns='Numerical Variable', values='Cramer V'
    )

    return association_matrix


def describe_associations(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    X_train_categorical = X_train.select_dtypes(include=['category'])
    X_train_numerical = X_train.select_dtypes(include=['number']).astype('float64')

    display(Markdown('## Feature Associations'))

    display(Markdown('### Categorical Feature Associations (Cramer V)'))
    display(_compute_cramer_v(X_train_categorical).round(2))

    display(Markdown('### Numerical Feature Correlations (Pearson)'))
    display(_compute_correlation(X_train_numerical).round(2))

    display(Markdown('### Categorical-Numerical Feature Associations (Cramer V)'))
    display(
        _compute_categorical_numerical_association(
            X_train_categorical,
            X_train_numerical.drop(columns=['capital.gain', 'capital.loss']),
        ).round(2)
    )

    display(Markdown('## Target Associations'))

    y_train_df = pd.DataFrame(y_train).rename(columns={y_train.name: 'target'})

    display(Markdown('### Categorical Features to Target (Cramer V)'))
    display(
        _compute_cramer_v(pd.concat([X_train_categorical, y_train_df], axis=1))
        .loc[:, 'target']
        .drop('target')
        .round(2)
    )

    display(Markdown('### Numerical Features to Target (Cramer V)'))
    display(
        _compute_categorical_numerical_association(
            y_train_df,
            X_train_numerical.drop(columns=['capital.gain', 'capital.loss']),
        ).round(2)
    )
