import pandas as pd
from pathlib import Path

raw_file = Path("data/raw/TeleCom_2009_2023_Daily.csv")
output_file = Path("data/processed/slt_daily_clean.csv")


def parse_suffix_number(x):
    """
    Convert strings like '9.90K', '1.2M', '3B', '123', '-', '' into floats.
    K = thousand, M = million, B = billion.
    Returns NaN for invalid/unparseable values.
    """
    if pd.isna(x):
        return pd.NA

    s = str(x).strip()

    if s in ("", "-", "—", "None", "nan"):
        return pd.NA

    # Remove commas
    s = s.replace(",", "")

    multiplier = 1.0
    last = s[-1].upper()

    if last == "K":
        multiplier = 1_000.0
        s = s[:-1]
    elif last == "M":
        multiplier = 1_000_000.0
        s = s[:-1]
    elif last == "B":
        multiplier = 1_000_000_000.0
        s = s[:-1]

    try:
        return float(s) * multiplier
    except ValueError:
        return pd.NA




def parse_percent_to_decimal(x):
    """
    Convert strings like '-0.53%' to -0.0053 (decimal form).
    Returns NaN for invalid values.
    """
    if pd.isna(x):
        return pd.NA

    s = str(x).strip()
    if s in ("", "-", "—", "None", "nan"):
        return pd.NA

    s = s.replace("%", "").replace(",", "")

    try:
        return float(s) / 100.0
    except ValueError:
        return pd.NA



def parse_special_columns(df: pd.DataFrame, volume_col: str = "volume", change_pct_col: str = "change_pct"):
    df = df.copy()

    if volume_col in df.columns:
        df[volume_col] = df[volume_col].apply(parse_suffix_number)

    if change_pct_col in df.columns:
        df[change_pct_col] = df[change_pct_col].apply(parse_percent_to_decimal)

    return df



def load_raw_data(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"CSV file not found!!!\n"
            f"searched file path: {file_path}"
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



def rename_slt_columns(df: pd.DataFrame):
    df = df.copy()

    rename_map = {
        "price": "close",
        "vol.": "volume",
        "change_%": "change_pct"
    }

    df = df.rename(columns=rename_map)
    return df



def parse_and_clean_date(df: pd.DataFrame, date_col: str = "date"):
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

    duplicate_date_count = df.duplicated(subset=[date_col]).sum()

    if duplicate_date_count > 0:
        print(f"Warning: dropping {duplicate_date_count} rows with duplicate entries. Keeping the last one.")
        df = df.drop_duplicates(subset=[date_col], keep="last").reset_index(drop=True)

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


def remove_invalid_rows(df:pd.DataFrame,open_col="open", high_col="high", low_col="low", close_col="close", volume_col="volume"):
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

    if not df.index.is_unique:
        raise ValueError(f"Date index is not unique. Search for duplicate date entries.")

    return df


def create_prediction_targets(df: pd.DataFrame, close_col: str = "close"):
    df = df.copy()

    if close_col not in df.columns:
        raise KeyError(f"'{close_col}' is not found in the dataframe columns.")
    
    df["target_next_close"] =  df[close_col].shift(-1)
    df["target_next_return"] = (df[close_col].shift(-1) - df[close_col]) / df[close_col]
    df["target_direction"] = (df["target_next_return"] > 0).astype(int)

    df = df.dropna(subset=["target_next_close", "target_next_return"])

    return df


def save_processed_data(df: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"\nSaved cleaned dataset to: {output_path}")


def main():
 
    df = load_raw_data(raw_file)

   
    inspect_data(df)

  
    df = standardize_column_names(df)
    df = rename_slt_columns(df)

    df = parse_and_clean_date(df, date_col="date")


    df = sort_and_remove_duplicate_dates(df, date_col="date")

    
    df = parse_special_columns(df, volume_col="volume", change_pct_col="change_pct")

   
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df = convert_numeric_columns(df, numeric_cols=numeric_cols)

    
    price_cols = ["open", "high", "low", "close"]
    df = handle_missing_values(df, price_cols=price_cols, volume_col="volume")

    
    df = remove_invalid_rows(df)

    
    df = set_date_index(df, date_col="date")

    
    df = create_prediction_targets(df, close_col="close")

    
    save_processed_data(df, output_file)

    
    print("---------- CLEANED DATA PREVIEW ----------")
    print(df.head())
    print("---------- CLEANED DATA INFO ----------")
    print(df.info())


if __name__ == "__main__":
    main()
