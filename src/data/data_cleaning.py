import pandas as pd
from pathlib import Path

raw_file = Path("data\raw\TeleCom_2009_2023_Daily.csv")
output_file = Path("data\processed\slt_daily_clean.csv")


def load_raw_data(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"CSV file not found!!!\n"
            "searched file path: {file_path}"
        )
    
    df = pd.read_csv(file_path)
    return df


def inspect_data(df: pd.DataFrame):
    print("----------Basic Information----------")
    print(df.info())

    print("----------Head----------")
    print(df.head())

    print("----------Tail----------")
    print(df.tail())

    print("----------Missing Values Per Columns----------")
    print(df.isna().sum())

    print("----------Duplicated row counts----------")
    print(df.duplicated().sum())


def standardize_column_names(df: pd.DataFrame):
    df = df.copy()
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ","_")
    )
    return df



def parse_and_clean(df: pd.DataFrame, date_col: str = "date"):
    df = df.copy()

    if date_col not in df.columns:
        raise KeyError(
            f"Date column is not found in the dataframe.\n"
            f"These are the found column names: {df.columns}"
        )
    
    df[date_col] = pd.to_datetime(df[date_col],errors="coerce")

    bad_dates_count = df[date_col].isna().sum()

    if bad_dates_count > 0:
        print(f"Warning: dropping {bad_dates_count} rows with invalid dates.")
        df = df.dropna(subset=[date_col])

    return df


def sort_and_remove_duplicate_dates(df:pd.DataFrame, date_col: str = "date"):
    df = df.copy()

    df = df.sort_values(date_col).reset_index(drop=True)

    duplicate_date_count = df.duplicated(subset=df[date_col]).sum()

    if duplicate_date_count > 0:
        print(f"Warning: dropping {duplicate_date_count} rows with duplicate entries. Keeping the last one.")
        df = df.duplicated(subset=df[date_col], keep="last").reset_index(drop=True)
    
    return df


def convert_numeric_columns(df:pd.DataFrame, numeric_cols: list[str]):
    df = df.copy()

    for col in numeric_cols:
        if col not in df.columns:
            print(f"{col} is not found in the dataframe columns. Skipping and moving to next one in the list")
            continue

        df[col] = df[col].astype(str).str.replace(",","",regex=False)   # removing commas in number 

        df[col] = pd.to_numeric(df[col],errors="coerce")


    return df


def handle_missing_values(df:pd.DataFrame, price_cols: list[str], volume_col: str = "volume"):
    # Prices -> forward filling method
    # Volume -> fill missings with 0

    df = df.copy()

    existing_price_cols = [col for col in price_cols if col in df.columns]

    if existing_price_cols:
        df[existing_price_cols] = df[existing_price_cols].ffill()
        df= df.dropna(subset=existing_price_cols)

    if volume_col in df.columns:
        df[volume_col] = df[volume_col].fillna(0)

    return df


def remove_invalid_rows(df:pd.DataFrame,open_col="open", high_col="high", low_col="low", close_col="price", volume_col="volume"):
    df = df.copy()

    #prices > 0
    for col in [open_col,high_col,low_col,close_col,volume_col]:
        if col in df.columns:
            df = df[df[col]>0]

    
    #high price >= low price
    if high_col in df.columns and low_col in df.columns:
        df = df[df[high_col]>=df[low_col]]

    if volume_col in df.columns:
        df = df[df[volume_col]>=0]

    return df.reset_index(drop=True)



def set_date_index(df:pd.DataFrame, date_col:str="date"):
    df = df.copy()
    df = df.set_index(keys=date_col)
    df = df.sort_index()

    if not df.index.unique():
        raise ValueError(f"Date index is not unique. Search for duplicate date entries.")

    return df





