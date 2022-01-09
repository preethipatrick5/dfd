import argparse
import os.path

import pandas as pd
import torch.utils.data
import torchmetrics
import tqdm

import train
from dfd.utils import experiments


def create_folder(folder_name):
    try:
        os.makedirs(folder_name)
    except:
        print("Results folder already exists")


def run_eval(model, exp_name, data_loader, results_folder, gpu):
    create_folder(results_folder)
    create_folder(os.path.join(results_folder, exp_name))
    mean = lambda x: sum(x) / len(x)
    lit_model = model.to("cuda:2")
    ys = []
    y_preds = []
    losses = []
    accuracies = []
    files = []
    with torch.no_grad():
        with tqdm.tqdm(data_loader) as its:
            for x, y, file in its:
                x = x.to(f"cuda:{gpu}")
                y = y.to(f"cuda:{gpu}")
                files.append(file)
                y_pred = lit_model(x)
                y_pred = y_pred.view(-1, )
                loss = torch.nn.functional.binary_cross_entropy(y_pred.type(torch.float32), y.type(torch.float32))
                accuracy = torchmetrics.functional.accuracy(torch.round(y_pred).type(torch.int32), y.type(torch.int32))
                losses.append(loss.detach().cpu())
                accuracies.append(accuracy.detach().cpu())
                ys += y.detach().cpu().numpy().tolist()
                y_preds += y_pred.detach().cpu().numpy().tolist()
                its.set_description(f"Loss : {mean(losses)}, Acc : {mean(accuracies)}")
        perf_file = os.path.join(results_folder, exp_name, f"{exp_name}_perf.txt")
        result_file = os.path.join(results_folder, exp_name, "result.pkl")
        with open(perf_file, "w") as file:
            file.write(f"Loss : {mean(losses)}, Acc : {mean(accuracies)}")
        pd.DataFrame({'file': files, "ys": ys, "y_preds": y_preds}).to_pickle(result_file)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--root", required=True)
    arg_parser.add_argument("--for_data", required=True)
    arg_parser.add_argument("--epochs", required=True)
    arg_parser.add_argument("--model", required=True)
    arg_parser.add_argument("--gpu", required=False, default=None)
    arg_parser.add_argument("--dev_run", required=False, default='0')
    arg_parser.add_argument("--train_type", required=False, default='clean')
    args = arg_parser.parse_args()
    for experiment_category in experiments.get_all_experiments():
        for experiment_name, experiment_transforms in experiment_category.items():
            model, data_module = train.get_trainable(args.model, args.for_data, args.root, args.train_type,
                                                     return_path=True, transforms=experiment_transforms,
                                                     from_checkpoint=True)
            results_folder = f"results/{args.model}"
            run_eval(model=model, exp_name=experiment_name, data_loader=data_module.val_dataloader(),
                     results_folder=results_folder, gpu=args.gpu)


if __name__ == '__main__':
    main()
