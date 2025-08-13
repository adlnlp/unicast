import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch

context_length = 56
forecast_length = 56

image_height = 384
image_width = 384

df = pd.read_csv("../csv/NN5/nn5.csv", header=None)
train_df, temp_df = train_test_split(df, test_size=0.4, shuffle=False)
val_df, test_df = train_test_split(temp_df, test_size=0.5, shuffle=False)

def create_dataset(df, split: str, forecast_length):
    scaler = StandardScaler()

    num_row = df.shape[0]
    num_col = df.shape[1]

    inputs = dict()
    targets = dict()

    for i in range(num_row-context_length-forecast_length):
        scaler.fit(df.values[i:i+context_length])
        values = scaler.transform(df.values[i:i+context_length+forecast_length])

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

print("train")
create_dataset(train_df, 'train', forecast_length)
print("val")
create_dataset(val_df, 'val', forecast_length)
print("test")
create_dataset(test_df, 'test', forecast_length)