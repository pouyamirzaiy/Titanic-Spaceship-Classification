import pandas as pd

def clean_data(df):
    # Fill missing values for numerical columns
    df.fillna({"Age": 29, "RoomService": 0, "FoodCourt": 0, "ShoppingMall": 0, "Spa": 0, "VRDeck": 0}, inplace=True)
    
    # Fill missing values for categorical columns
    df.fillna({"HomePlanet": "Earth", "CryoSleep": False, "Destination": "TRAPPIST-1e", "VIP": False}, inplace=True)
    
    # Drop the 'Name' column
    df.drop(columns="Name", inplace=True)
    
    # Handle missing values in 'Cabin' columns
    values = df[df["Deck"].isnull()]["Group"].tolist()
    count = 0
    for value in values:
        filtered = df[df["Group"] == value][["Deck", "Num", "Side"]].dropna()
        if not filtered.empty:
            deck = filtered["Deck"].unique()
            num = filtered["Num"].unique()
            side = filtered["Side"].unique()
            if not len(deck) > 1:
                rows = df[df["Group"] == value]
                rows_nan = rows[rows["Deck"].isna()].index
                rows.loc[rows_nan, "Deck"] = deck
                rows.loc[rows_nan, "Num"] = num
                rows.loc[rows_nan, "Side"] = side
                df.loc[df["Group"] == value] = rows
                count += 1
    df.dropna(subset=["Deck", "Num", "Side"], inplace=True)
    print(f"The number of values cleaned is: {count}")
    
    return df

if __name__ == "__main__":
    df = pd.read_csv('/content/train.csv')
    df = clean_data(df)
    df.to_csv('/content/cleaned_train.csv', index=False)
