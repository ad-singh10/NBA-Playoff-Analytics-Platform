import pandas as pd

games = pd.read_csv("data/raw/games.csv")

print(games.columns.to_list())