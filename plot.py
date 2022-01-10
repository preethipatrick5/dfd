import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt

root = "/home/ram/Downloads/results"


model_map = {
    "resnet": "res",
    "xception": "xcptn",
    "resnet3": "res3d",
    "resnet_lstm": "reslstm"
}

experiment_map = {
    "padding": "pad",
    "gaussian_blur": "gaus_blur",
    "gaussian_noise": "gaus_noise",
    "random_noise": "rand_noise",
    "baseline": "baseline",
    "shuffle_pixels": "shuf_pixel",
    "saturation": "sat",
    "perspective_transform": "pers_trans",
    "random_augmentation": "mixed_aug",
    "pixelization": "pixel"
}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def is_parametered_experiment(experiment):
    experiment = experiment.split("_")[-1]
    try:
        float(experiment)
        return True
    except:
        return False


rocs = []
datasets = ["dfdc", "ff"]
trainings = ["clean", "noise"]
models = ["resnet", "xception", "resnet_lstm", "resnet3"]
for for_data in datasets:
    for for_training in trainings:
        for for_model in models:
            for experiment in os.listdir(os.path.join(root, for_data, for_training, for_model)):
                if os.path.isfile(os.path.join(root, for_data, for_training, for_model, experiment)):
                    continue
                if is_parametered_experiment(experiment):
                    continue
                else:
                    file_path = os.path.join(root, for_data, for_training, for_model, experiment, "result.pkl")
                    data = pd.read_pickle(file_path)
                    y = data["ys"].to_numpy()
                    y_pred = data["y_preds"].to_numpy()
                    score = roc_auc_score(y, y_pred)
                    fpr, tpr, thresholds = roc_curve(y, y_pred)
                    rocs.append({
                        "model": for_model,
                        "experiment": experiment,
                        "fpr": fpr,
                        "tpr": tpr,
                        "score": score,
                        "data_state": for_training,
                        "dataset": for_data
                    })
        print(f"{for_data} - {for_training} done")
    print(f"{for_data} done")
data = pd.DataFrame(rocs)
print(data.head())

for for_data in datasets:
    for for_training in trainings:
        for experiment in os.listdir(os.path.join(root, for_data, "noise", "resnet")):
            plt.figure(figsize=(6, 4))
            experiment_name = "mixed_augmentation" if experiment == "random_augmentation" else experiment
            for for_model in models:
                display_name = ""
                if for_model in ["resnet", "xception"]:
                    display_name = f"CNN_{for_model}"
                elif for_model == "resnet_lstm":
                    display_name = "CNN+LSTM"
                elif for_model == "resnet3":
                    display_name = "3D-CNN"
                for index, row in data[(data['model'] == for_model) & (data['data_state'] == for_training) & (
                        data['dataset'] == for_data) & (data['experiment'] == experiment)].iterrows():
                    fpr = row['fpr']
                    tpr = row['tpr']
                    if for_model in ["resnet_lstm"]:
                        fpr = gaussian_filter1d(fpr, sigma=10)
                        tpr = gaussian_filter1d(tpr, sigma=10)
                    score = row['score']
                    plt.plot(fpr, tpr, label=f"{display_name}-{score}")
                    plt.xlabel("True Positive Rate")
                    plt.ylabel("False Positive Rate")
                    plt.legend(loc=4)
            plt.savefig(f"graphs/roc_{for_data}_{for_training}_{experiment_name}.png")
    plt.close()
plt.legend(loc=4)
# plt.show()
