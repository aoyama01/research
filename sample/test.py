# %%
import pandas as pd
from icecream import ic
from matplotlib import pyplot as plt

data = pd.read_csv(
    "C:/Users/Shunya Aoyama/OneDrive - OUMail (Osaka University)/B4_AW/GradRes/code/by-kiyono/EEG_RRI結合/EEG_RRI.csv"
)

plt.plot(data["meanRR"][200:])
ic(data["meanRR"][200:])
