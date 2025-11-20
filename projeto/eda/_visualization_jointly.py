import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display


def _visualize_categorical_vs_numerical(
    categorical_data: pd.Series,
    numerical_data: pd.Series,
    title: str,
) -> None:
    display(Markdown(f'### {title}'))

    data = pd.DataFrame(
        {
            'categorical': categorical_data,
            'numerical': numerical_data,
        }
    ).dropna()

    plt.figure(figsize=(10, 6))
    data.boxplot(column='numerical', by='categorical', grid=False)
    plt.title(title)
    plt.suptitle('')
    plt.xlabel('Category')
    plt.ylabel('Numerical Value')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def _visualize_numerical_vs_numerical(
    x_data: pd.Series,
    y_data: pd.Series,
    title: str,
) -> None:
    display(Markdown(f'### {title}'))

    data = pd.DataFrame(
        {
            'x': x_data,
            'y': y_data,
        }
    ).dropna()

    plt.figure(figsize=(8, 6))
    plt.scatter(data['x'], data['y'], alpha=0.6)
    plt.title(title)
    plt.xlabel('X Value')
    plt.ylabel('Y Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def _visualize_categorical_vs_categorical(
    categorical_data_1: pd.Series,
    categorical_data_2: pd.Series,
    title: str,
) -> None:
    display(Markdown(f'### {title}'))

    data = pd.DataFrame(
        {
            'cat1': categorical_data_1,
            'cat2': categorical_data_2,
        }
    ).dropna()

    contingency_table = pd.crosstab(data['cat1'], data['cat2'])

    plt.figure(figsize=(10, 6))
    contingency_table.plot(kind='barh', stacked=True, colormap='viridis')
    plt.title(title)
    plt.xlabel('Category 1')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Category 2', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def visualize_jointly(
    X_train: pd.DataFrame, y_train: pd.Series, show_joint_features: bool = False
) -> None:
    X_train_categorical = X_train.select_dtypes(include=['object', 'category'])
    categorical_cols = X_train_categorical.columns

    X_train_numerical = X_train.select_dtypes(include=['number']).astype('float64')
    numerical_cols = X_train_numerical.columns

    display(Markdown('## Feature vs. Target Visualizations'))

    for cat_col in categorical_cols:
        _visualize_categorical_vs_categorical(
            X_train_categorical[cat_col],
            y_train,
            title=f'Relationship between {cat_col} and Target',
        )
    for num_col in numerical_cols:
        _visualize_categorical_vs_numerical(
            y_train,
            X_train_numerical[num_col],
            title=f'Relationship between {num_col} and Target',
        )

    if not show_joint_features:
        return

    display(Markdown('## Feature vs. Feature Visualizations'))

    # Categorical vs. Categorical
    for col_1 in categorical_cols:
        for col_2 in categorical_cols:
            if col_1 >= col_2:
                continue
            _visualize_categorical_vs_categorical(
                X_train_categorical[col_1],
                X_train_categorical[col_2],
                title=f'Relationship between {col_1} and {col_2}',
            )

    # Numerical vs. Numerical
    for col_1 in numerical_cols:
        for col_2 in numerical_cols:
            if col_1 >= col_2:
                continue
            _visualize_numerical_vs_numerical(
                X_train_numerical[col_1],
                X_train_numerical[col_2],
                title=f'Relationship between {col_1} and {col_2}',
            )

    # Categorical vs. Numerical
    for cat_col in categorical_cols:
        for num_col in numerical_cols:
            _visualize_categorical_vs_numerical(
                X_train_categorical[cat_col],
                X_train_numerical[num_col],
                title=f'Relationship between {cat_col} and {num_col}',
            )
