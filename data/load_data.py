import pandas as pd
from sklearn.model_selection import train_test_split
import ast
from typing import List, Union
from datetime import timedelta, date

def impute_missing_dates(dates: List[Union[date,None]]) -> List[date]:
    if all(d is None for d in dates):
        return [date(2024, 1, 1) for _ in dates]
    elif all(d is not None for d in dates):
        return dates

    n = len(dates)
    result = dates.copy()

    # Fill missing in middle using avg of left and right
    for i in range(n):
        if result[i] is None:
            # find left
            left = next((result[j] for j in range(i - 1, -1, -1) if result[j] is not None), None)
            # find right
            right = next((result[j] for j in range(i + 1, n) if result[j] is not None), None)

            if left and right:
                avg_seconds = (left.timestamp() + right.timestamp()) / 2
                result[i] = date.fromtimestamp(avg_seconds)
            elif left and not right:
                result[i] = left + timedelta(days=1)
            elif right and not left:
                result[i] = right - timedelta(days=1)
            else:
                result[i] = date(2024, 1, 1)

    return result
def pop_dates(visits: List[str]) -> pd.Series[List[str], List[date]]:
    new_visits = []
    dates = []
    for visit in visits:
        new_visit = visit.split("Fecha: ")[1]
        try:
            date_str, new_visit = new_visit[:9], new_visit[9:]
            d = date.strptime(date_str, "%d%b%Y")
        except ValueError:
            d = None
        new_visits.append(new_visit)
        dates.append(d)
    return pd.Series([new_visits, impute_missing_dates(dates)])

def load_real_data(path, stratify_col='label', test_size=0.2, dev_size=0.1, max_visits=40, no_dig=False):
    df = pd.read_csv(path)
    if no_dig:
        df = df[df['label']!=2]
    df['visits'] = df['visits'].apply(ast.literal_eval)
    df['visits'] = df.apply(lambda row: row['visits'] if row['num_visits'] <= max_visits else row['visits'][-40:], axis=1) # if there are more than 40 visits, take only the last 40
    df[['visits', 'dates']] = df['visits'].apply(pop_dates)
    df.rename(columns={'clasificacion_diag': 'label'}, inplace=True)
    df['num_visits'] = df['visits'].apply(len) # recompute number of visits for cases with more than 40 visits
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df[stratify_col])
    train_df, dev_df = train_test_split(train_df, test_size=dev_size, random_state=42, stratify=train_df[stratify_col])
    return train_df, test_df, dev_df