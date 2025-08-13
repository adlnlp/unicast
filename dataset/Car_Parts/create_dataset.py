import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch

context_length = 12
forecast_length = 12

image_height = 384
image_width = 384

df = pd.read_csv("../csv/Car_Parts/car_parts.csv") # 51 * 2674
df = df.drop(columns="timestamp")
shuffled_df = df.sample(frac=1, axis=1)
scaler = StandardScaler()
shuffled_df = scaler.fit_transform(shuffled_df)
train_df = shuffled_df[:, :1604]
val_df = shuffled_df[:, 1604:1604+535]
test_df = shuffled_df[:, 1604+535:]

def create_dataset(df, split: str, forecast_length):
    num_row, num_col = df.shape

    inputs = dict()
    targets = dict()

    for i in range(num_row-context_length-forecast_length):
        values = df[i:i+context_length+forecast_length]
        for j in range(num_col):
            if not os.path.exists(f"{split}/img/{j}_{i}.png"):
                plt.figure(figsize=(image_width/100, image_height/100), dpi=100)
                plt.plot(values[:context_length, j], color="black", linestyle="-", linewidth=1, marker="*", markersize=1)
                plt.xticks([])
                plt.yticks([])
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0,0)
                plt.savefig(f"{split}/img/{j}_{i}.png", pad_inches=0)
                plt.close()

            input = values[:context_length, j]
            inputs[f'{j}_{i}.png'] = torch.tensor(input).unsqueeze(dim=0)
            target = values[context_length:context_length+forecast_length, j]
            targets[f'{j}_{i}.png'] = torch.tensor(target).unsqueeze(dim=0)
    torch.save(inputs, f'{split}/inputs.pt')
    torch.save(targets, f'{split}/targets_{forecast_length}.pt')

print(forecast_length)
print("train")
create_dataset(train_df, 'train', forecast_length)
print("val")
create_dataset(val_df, 'val', forecast_length)
print("test")
create_dataset(test_df, 'test', forecast_length)