import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display


def _visualize_age(dataframe: pd.DataFrame) -> None:
    display(Markdown('### Age Distribution'))

    plt.figure(figsize=(8, 6))
    plt.hist(
        dataframe['age'].dropna(),
        bins=np.arange(0, 100, 1).tolist(),
        color='skyblue',
        edgecolor='black',
    )
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()


def _visualize_education(dataframe: pd.DataFrame) -> None:
    display(Markdown('### Education Level Distribution'))

    education_counts = dataframe['education'].value_counts().sort_index()

    plt.figure(figsize=(8, 6))
    education_counts.plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.title('Education Level Distribution')
    plt.xlabel('Education Level')
    plt.ylabel('Number of Individuals')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()


def _visualize_education_num(dataframe: pd.DataFrame) -> None:
    display(Markdown('### Education Number Distribution'))

    plt.figure(figsize=(8, 6))
    plt.hist(
        dataframe['education.num'].dropna(),
        bins=np.arange(0, dataframe['education.num'].max() + 1, 1).tolist(),
        color='salmon',
        edgecolor='black',
    )
    plt.title('Education Number Distribution')
    plt.xlabel('Education Number')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()


def _visualize_marital_status(dataframe: pd.DataFrame) -> None:
    display(Markdown('### Marital Status Distribution'))

    marital_status_counts = dataframe['marital.status'].value_counts().sort_index()

    plt.figure(figsize=(8, 6))
    marital_status_counts.plot(kind='bar', color='orchid', edgecolor='black')
    plt.title('Marital Status Distribution')
    plt.xlabel('Marital Status')
    plt.ylabel('Number of Individuals')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()


def _visualize_occupation(dataframe: pd.DataFrame) -> None:
    display(Markdown('### Occupation Distribution'))

    occupation_counts = dataframe['occupation'].value_counts().sort_values()

    plt.figure(figsize=(8, 6))
    occupation_counts.plot(kind='barh', color='gold', edgecolor='black')
    plt.title('Occupation Distribution')
    plt.xlabel('Number of Individuals')
    plt.ylabel('Occupation')
    plt.grid(axis='x', alpha=0.75)
    plt.tight_layout()
    plt.show()


def _visualize_relationship(dataframe: pd.DataFrame) -> None:
    display(Markdown('### Relationship Distribution'))

    relationship_counts = dataframe['relationship'].value_counts().sort_values()

    plt.figure(figsize=(8, 6))
    relationship_counts.plot(kind='bar', color='cyan', edgecolor='black')
    plt.title('Relationship Distribution')
    plt.xlabel('Relationship')
    plt.ylabel('Number of Individuals')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()


def _visualize_race(dataframe: pd.DataFrame) -> None:
    display(Markdown('### Race Distribution'))

    race_counts = dataframe['race'].value_counts().sort_values()

    plt.figure(figsize=(8, 6))
    race_counts.plot(kind='bar', color='lightcoral', edgecolor='black')
    plt.title('Race Distribution')
    plt.xlabel('Race')
    plt.ylabel('Number of Individuals')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    race_counts.plot(kind='bar', color='lightcoral', edgecolor='black')
    plt.title('Race Distribution - log scale')
    plt.xlabel('Race')
    plt.ylabel('Number of Individuals')
    plt.yscale('log')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()


def _visualize_sex(dataframe: pd.DataFrame) -> None:
    display(Markdown('### Sex Distribution'))

    sex_counts = dataframe['sex'].value_counts().sort_values()

    plt.figure(figsize=(8, 6))
    sex_counts.plot(kind='bar', color='lightblue', edgecolor='black')
    plt.title('Sex Distribution')
    plt.xlabel('Sex')
    plt.ylabel('Number of Individuals')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()


def _visualize_capital_gain(dataframe: pd.DataFrame) -> None:
    display(Markdown('### Capital Gain Distribution'))

    plt.figure(figsize=(8, 6))
    plt.hist(
        dataframe['capital.gain'].dropna(),
        bins=np.arange(0, dataframe['capital.gain'].max() + 1, 100).tolist(),
        color='lightyellow',
        edgecolor='black',
    )
    plt.title('Capital Gain Distribution')
    plt.xlabel('Capital Gain')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()


def _visualize_capital_loss(dataframe: pd.DataFrame) -> None:
    display(Markdown('### Capital Loss Distribution'))

    plt.figure(figsize=(8, 6))
    plt.hist(
        dataframe['capital.loss'].dropna(),
        bins=np.arange(0, dataframe['capital.loss'].max() + 1, 100).tolist(),
        color='lightpink',
        edgecolor='black',
    )
    plt.title('Capital Loss Distribution')
    plt.xlabel('Capital Loss')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()


def _visualize_hours_per_week(dataframe: pd.DataFrame) -> None:
    display(Markdown('### Hours Per Week Distribution'))

    plt.figure(figsize=(8, 6))
    plt.hist(
        dataframe['hours.per.week'].dropna(),
        bins=np.arange(0, 100, 1).tolist(),
        color='lightgrey',
        edgecolor='black',
    )
    plt.title('Hours Per Week Distribution')
    plt.xlabel('Hours Per Week')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(
        dataframe['hours.per.week'].dropna(),
        bins=np.arange(0, 100, 5).tolist(),
        color='lightgrey',
        edgecolor='black',
    )
    plt.title('Hours Per Week Distribution - 5 hour bins')
    plt.xlabel('Hours Per Week')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(
        dataframe['hours.per.week'].dropna(),
        bins=np.arange(0, 100, 1).tolist(),
        color='lightgrey',
        edgecolor='black',
    )
    plt.title('Hours Per Week Distribution - log scale')
    plt.xlabel('Hours Per Week')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(
        dataframe['hours.per.week'].dropna(),
        bins=np.arange(0, 100, 5).tolist(),
        color='lightgrey',
        edgecolor='black',
    )
    plt.title('Hours Per Week Distribution - 5 hour bins - log scale')
    plt.xlabel('Hours Per Week')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()


def _visualize_native_country(dataframe: pd.DataFrame) -> None:
    display(Markdown('### Native Country Distribution'))

    native_country_counts = dataframe['native.country'].value_counts().sort_values()

    plt.figure(figsize=(8, 8))
    native_country_counts.plot(kind='barh', color='lightseagreen', edgecolor='black')
    plt.title('Native Country Distribution')
    plt.xlabel('Number of Individuals')
    plt.ylabel('Native Country')
    plt.grid(axis='x', alpha=0.75)
    plt.tight_layout()
    plt.show()


def _visualize_income(dataframe: pd.DataFrame) -> None:
    display(Markdown('### Income Distribution'))

    income_counts = dataframe['income'].value_counts().sort_index()

    plt.figure(figsize=(8, 6))
    income_counts.plot(kind='bar', color='plum', edgecolor='black')
    plt.title('Income Distribution')
    plt.xlabel('Income')
    plt.ylabel('Number of Individuals')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()


def visualize_univariate(dataframe: pd.DataFrame) -> None:
    display(Markdown('## Visualizations'))
    _visualize_age(dataframe)
    _visualize_education(dataframe)
    _visualize_education_num(dataframe)
    _visualize_marital_status(dataframe)
    _visualize_occupation(dataframe)
    _visualize_relationship(dataframe)
    _visualize_race(dataframe)
    _visualize_sex(dataframe)
    _visualize_capital_gain(dataframe)
    _visualize_capital_loss(dataframe)
    _visualize_hours_per_week(dataframe)
    _visualize_native_country(dataframe)
    _visualize_income(dataframe)
