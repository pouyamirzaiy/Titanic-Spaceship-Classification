import pandas as pd
from sklearn.preprocessing import LabelEncoder

def feature_engineering(df):
    encoder = LabelEncoder()
    
    df["Group"] = encoder.fit_transform(df["Group"])
    df["Deck"] = encoder.fit_transform(df["Deck"])
    df["Transported"] = encoder.fit_transform(df["Transported"])
    df["CryoSleep"] = df["CryoSleep"].replace({False: 0, True: 1})
    df["VIP"] = df["VIP"].replace({False: 0, True: 1})
    
    homeplanet = pd.get_dummies(df["HomePlanet"], dtype=int)
    side = pd.get_dummies(df["Side"], dtype=int)
    destination = pd.get_dummies(df["Destination"], dtype=int)
    
    df.drop(columns=["HomePlanet", "Side", "Destination"], inplace=True)
    
    df_final = pd.concat([df, homeplanet, side, destination], axis=1, ignore_index=True)
    df_final.columns = ['Group', 'Member', 'CryoSleep', 'Deck', 'Num', 'Age', 'VIP', 'RoomService', 'FoodCourt',
                        'ShoppingMall', 'Spa', 'VRDeck', 'Transported', 'Earth', 'Europa', 'Mars', 'Port', 'Starboard',
                        '55 Cancri e', 'PSO J318.5-22', 'TRAPPIST-1e']
    
    df_final.drop(columns=['Num'], inplace=True)
    
    return df_final

if __name__ == "__main__":
    df = pd.read_csv('/content/cleaned_train.csv')
    df_final = feature_engineering(df)
    df_final.to_csv('/content/engineered_train.csv', index=False)
