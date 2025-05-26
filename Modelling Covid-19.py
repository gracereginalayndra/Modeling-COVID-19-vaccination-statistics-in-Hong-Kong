import pandas as pd
import matplotlib.pyplot as plt
import json
import requests
import matplotlib.ticker as mticker


# Globals
N = 7  # number of dose columns for each type of vaccine
COLOR = [
    "#fbc58d", "#7caedb", "#99fe66", "#82b869", "#257252", "#ffbf00", "#ec7d31"
]


def get_population(start: str, end: str) -> pd.DataFrame:
    """Download population data from gov.hk data API into a data frame."""
    url = "https://www.censtatd.gov.hk/api/post.php"
    data = pd.read_json(url)
    parameters = {
        "cv": {
            "SEX": [""],
            "AGE": ["zero"] + [str(i) for i in range(1, 85)] + ["85+"]
        },
        "sv": {
            "POP": ["Raw_K_1dp_per_n"]
        },
        "period": {
            "start": start,
            "end": end
        },
        "id": "110-01002",
        "lang": "en"
    }
    data = {'query': json.dumps(parameters)}
    r = requests.post(url, data=data, timeout=20)
    return pd.DataFrame(r.json()["dataSet"])


def preprocess_population(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess population DataFrame to categorize age groups
    and aggregate figures.

    Parameters:
    - df (pd.DataFrame): DataFrame containing population data.

    Returns:
    - pd.DataFrame: Preprocessed DataFrame with age groups
    and aggregated figures.

    This function preprocesses the population DataFrame by:
    - Dropping the first row containing empty age.
    - Converting age values to integers.
    - Categorizing age groups and aggregating figures.
    """
    # Carry out data preprocessing for the population data frame.

    df = df.copy()  # This can avoid SettingWithCopyWarning

    # Drop the first row containing empty age
    df = df.drop(0)

    # Replace "zero" and "85+" with appropriate values for type conversion
    df['AGE'] = df['AGE'].replace({'zero': '0', '85+': '85'})

    # Convert AGE column to integers
    df['AGE'] = df['AGE'].astype(int)

    # Set the bins and labels for age groups
    bins = [float("-inf"), 11, 19, 29, 39, 49, 59, 69, 79, float("inf")]
    labels = ['0-11', '12-19', '20-29', '30-39', '40-49',
              '50-59', '60-69', '70-79', '80 and above']

    # Create the Age Group column based on the AGE values
    df['Age Group'] = pd.cut(df['AGE'], bins=bins, labels=labels)

    # Group by Age Group and calculate the sum of figure values
    df_grouped = df.groupby('Age Group',
                            observed=True)['figure'].sum().reset_index()

    # Convert the figure column to int64
    df_grouped['figure'] = (df_grouped['figure']*1000).astype('int64')

    # Set Age Group as the index
    df_grouped = df_grouped.set_index('Age Group')

    # Return the preprocessed data frame
    return df_grouped


def get_vaccination(start: str = None, end: str = None) -> pd.DataFrame:
    """
    Download vaccination data from the gov.hk data API and
    filter it based on specified date range.

    Parameters:
    - start (str, optional): Start date of the date range to
      filter the data. Defaults to None.
    - end (str, optional): End date of the date range to
      filter the data. Defaults to None.

    Returns:
    - pd.DataFrame: DataFrame containing vaccination data
      within the specified date range.

    This function downloads vaccination data from the Hong Kong
    government's data API and filters it based on the specified
    start and end dates. It then drops the 'Sex' column,
    converts the 'Date' column to datetime format,
    and sets the 'Date' column as the index.
    """
    # Download vaccination data from gov.hk data API into a data frame.

    url = "https://www.healthbureau.gov.hk/download/opendata/COVID19" \
          + "/vaccination-rates-over-time-by-age.csv"

    df = pd.read_csv(url)
    df = df.drop(['Sex'], axis=1)
    # Convert the dates column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    max_date = df['Date'].max()
    min_date = df['Date'].min()

    start_date = start or min_date
    end_date = end or max_date

    filtered_df = df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    filtered_df = filtered_df.set_index('Date')
    return filtered_df


def preprocess_vaccination(df_vac: pd.DataFrame,
                           df_pop: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data preprocessing for vaccination data.

    Parameters:
    - df_vac (pd.DataFrame): DataFrame containing vaccination data.
    - df_pop (pd.DataFrame): DataFrame containing population data.

    Returns:
    - pd.DataFrame: Preprocessed DataFrame with vaccination rates by age group.

    This function preprocesses vaccination data by:
    - Grouping by age group and summing vaccine doses.
    - Calculating vaccination rates per age group.
    - Merging with population data.
    - Renaming columns for clarity.
    """
    # Carry out data preprocessing for the vaccination data frame.
    # Group by Age Group and calculate the sum of figure values
    list1 = list(df_vac.columns)
    df_vac = df_vac.groupby('Age Group',
                            observed=True)[list1[1:]].sum().reset_index()
    df_vac = df_vac.set_index('Age Group')
    # Set Age Group as the index
    for i in range(1, N+1):
        df_vac[f"{i}th vaccine dose"] = df_vac[list1[i]] + df_vac[list1[N+i]]
    # yielding wrong output
    # how can I make N global variable where N stores
    # the number of the vaccine, so do I pass the N as the argument?
    df_vac = df_vac.drop(list1[1:], axis=1)
    df_merged = pd.merge(df_vac, df_pop, on='Age Group')
    list2 = list(df_merged.columns)
    for i in range(0, N):
        df_merged[str(i+1)+'th vaccine dose'] = (df_merged[list2[i]]
                                                 / df_merged['figure'])*100
    df_merged = df_merged.drop(['figure'], axis=1)
    df_merged = df_merged.rename(columns={"1th vaccine dose":
                                          "1st vaccine dose",
                                          "2th vaccine dose":
                                          "2nd vaccine dose",
                                          "3th vaccine dose":
                                          "3rd vaccine dose"})

    return df_merged


def plot_vaccination(df: pd.DataFrame, as_at: str, n: int = N) -> plt.Figure:
    """
    Create a horizontal bar plot of vaccination data.

    Parameters:
    - df (pd.DataFrame): DataFrame containing vaccination data.
    - as_at (str): Date of the vaccination data.
    - n (int, optional): Number of vaccine doses to plot. Defaults to N.

    Returns:
    - plt.Figure: Figure object containing the bar plot.

    This function creates a bar plot of the vaccination data,
    showing the rates of vaccine doses taken in Hong Kong per
    population by age group.
    It customizes the plot appearance by labeling each bar
    with the corresponding percentage value and sets
    appropriate axis labels and title.
    """
    # Create a barh plot of n series of the vaccination data frame.

    # reverse row order to show smallest age group top

# Create the figure and axes instances

    with plt.rc_context({'font.size': 7}):
        fig, ax = plt.subplots(figsize=(10, 5))
        df_slice = df.iloc[:, :n].iloc[::-1]
        # Slice dataframe and reverse row order

        for i, (col, color) in enumerate(zip(df_slice.columns, COLOR)):
            suffix = "th"
            if i == 0:
                suffix = "st"
            elif i == 1:
                suffix = "nd"
            elif i == 2:
                suffix = "rd"
            ax.barh(df_slice.index, df_slice[col], color=color,
                    label=f"{i+1}{suffix} vaccine dose",
                    height=0.8, zorder=2)

            # Add data labels
            for j, val in enumerate(df_slice[col]):
                if i == 0:
                    ax.text(val + 0.5, j, f"{val:.1f}%", ha='left',
                            va='center', fontweight='bold')
                if i < len(df_slice.columns
                           ) - 1 and val - df_slice.iloc[j, i + 1] >= 5:
                    xpos = ((val - df_slice.iloc[j, i + 1])
                            / 2) + df_slice.iloc[j, i + 1]
                    ax.text(xpos, j, f"{val:.1f}%", ha='center', va='center')
                elif i >= len(df_slice.columns
                              ) - 1 and val >= 5:  # for ijo tua
                    ax.text(val / 2, j, f"{val:.1f}%",
                            ha='center', va='center')

        ax.set_title("Rate of vaccine doses taken in Hong Kong per population"
                     f" by age group, as at {as_at}", weight="bold")
        ax.set_xlabel("Per population rate")
        ax.set_ylabel("Age groups")
        ax.set_xlim([0, 115])
        ax.grid(axis='x', zorder=0)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        # Set x-axis tick formatter to display as percentage
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
        ax.set_xticks(range(0, 101, 10))

        plt.tight_layout()

    return fig

# Sample client code


if __name__ == "__main__":
    df_pop1 = get_population("202207", "202212")
    df_pop1 = preprocess_population(df_pop1)
    df_vac1 = get_vaccination(end="2023-01-29")
    df_vac1 = preprocess_vaccination(df_vac1, df_pop1)
    fig1 = plot_vaccination(df_vac1, as_at="2023-01-29", n=5)
    fig1.savefig("q5_fig1.png")

    df_pop2 = get_population("202307", "202312")
    df_pop2 = preprocess_population(df_pop2)
    df_vac2 = get_vaccination(end="2024-04-07")
    df_vac2 = preprocess_vaccination(df_vac2, df_pop2)
    fig2 = plot_vaccination(df_vac2, as_at="2024-04-07", n=6)
    fig2.savefig("q5_fig2.png")

    df_pop3 = df_pop2
    df_vac3 = get_vaccination()
    df_vac3 = preprocess_vaccination(df_vac3, df_pop3)
    fig3 = plot_vaccination(df_vac3, as_at="2024-04-07 (up to 7th-dose)")
    fig3.savefig("q5_fig3.png")
