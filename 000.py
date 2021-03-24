import pandas as pd
from config import config

text_df = pd.read_csv(config.data_path)

print(text_df)