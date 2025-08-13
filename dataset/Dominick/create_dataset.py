import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch

context_length = 8
forecast_length = 8

image_height = 384
image_width = 384

df = pd.read_csv("../csv/Dominick/dominick.csv")
df = df.drop(columns="timestamp")
shuffled_df = df.sample(frac=1, axis=1)
shuffled_df = shuffled_df.iloc[:, :100]
row, col = shuffled_df.shape

inputs = {
    "train": dict(),
    "val": dict(),
    "test": dict()
}
targets = {
    "train": dict(),
    "val": dict(),
    "test": dict()
}

for j in range(col):
    series = df.iloc[:, j].dropna()
    series_dict = dict()
    mean = series[:int(series.size*0.6)].mean()
    std = series[:int(series.size*0.6)].std()
    series_dict["train"] = ((series[:int(series.size*0.6)]-mean)/std).to_numpy()
    series_dict["val"] = ((series[int(series.size*0.6):int(series.size*0.6)+int(series.size*0.2)]-mean)/std).to_numpy()
    series_dict["test"] = ((series[int(series.size*0.6)+int(series.size*0.2):]-mean)/std).to_numpy()
    for split in ["train", "val", "test"]:
        s = series_dict[split]
        for i in range(s.size-context_length-forecast_length):
            values = s[i:i+context_length+forecast_length]

            if not os.path.exists(f"{split}/img/{j}_{i}.png"):
                plt.figure(figsize=(image_width/100, image_height/100), dpi=100)
                plt.plot(values[:context_length], color="black", linestyle="-", linewidth=1, marker="*", markersize=1)
                plt.xticks([])
                plt.yticks([])
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0,0)
                plt.savefig(f"{split}/img/{j}_{i}.png", pad_inches=0)
                plt.close()

            input = values[:context_length]
            inputs[split][f'{j}_{i}.png'] = torch.tensor(input).unsqueeze(dim=0)
            target = values[context_length:context_length+forecast_length]
            targets[split][f'{j}_{i}.png'] = torch.tensor(target).unsqueeze(dim=0)

for split in ["train", "val", "test"]:
    torch.save(inputs[split], f'{split}/inputs.pt')
    torch.save(targets[split], f'{split}/targets_{forecast_length}.pt')


# def create_dataset(df, split: str, forecast_length):
#     scaler = StandardScaler()

#     num_col = df.shape[1]

#     inputs = dict()
#     targets = dict()

#     for j in range(num_col):
#         series = df.iloc[:, j].dropna()
#         for i in range(series.size-context_length-forecast_length):
#             values = ((series[i:i+context_length+forecast_length] - series[i:i+context_length].mean())/series[i:i+context_length].std()).to_numpy()

#             if not os.path.exists(f"{split}/img/{j}_{i}.png"):
#                 plt.figure(figsize=(image_width/100, image_height/100), dpi=100)
#                 plt.plot(values[:context_length], color="black", linestyle="-", linewidth=1, marker="*", markersize=1)
#                 plt.xticks([])
#                 plt.yticks([])
#                 plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
#                 plt.margins(0,0)
#                 plt.savefig(f"{split}/img/{j}_{i}.png", pad_inches=0)
#                 plt.close()

#             input = values[:context_length]
#             inputs[f'{j}_{i}.png'] = torch.tensor(input).unsqueeze(dim=0)
#             target = values[context_length:context_length+forecast_length]
#             targets[f'{j}_{i}.png'] = torch.tensor(target).unsqueeze(dim=0)
#     torch.save(inputs, f'{split}/inputs.pt')
#     torch.save(targets, f'{split}/targets_{forecast_length}.pt')


# print("train")
# create_dataset(train_df, 'train', forecast_length)
# print("val")
# create_dataset(val_df, 'val', forecast_length)
# print("test")
# create_dataset(test_df, 'test', forecast_length)