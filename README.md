# Steps

### Environment setup
* Install requirements
```shell
pip install -r requirements.txt 
```
* Download datasets
  * [DFDC Preview Dataset](https://ai.facebook.com/datasets/dfdc/)
  * [FF++](https://www.kaggle.com/sorokin/faceforensics)

### Training
To train the models on DFDC dataset use
```shell
bash train_all.sh dfdc $path_to_dataset_folder $gpu $num_epochs $is_dev_run $train_for
```
path_to_dataset_folder is the root folder of the dfdc dataset, gpu is the id of the gpu device on which training is to be done, num_epochs is the number of epochs to train the model for, is_dev_run takes a boolean value. If it is true then the model is passed through one batch for training and one batch for testing. train_for takes 2 possible values either clean or noise. If the value of train_for is clean then the frames of the video are not exposed to any noise, and if the value is noise, then the frames of the video are exposed to noise before using for training.  


To train the model on FF++ dataset
```shell
bash train_all.sh ff $path_to_dataset_folder $num_epochs $is_dev_run $train_for
```

### Evaluation
```shell
bash eval_all.sh dfdc $path_to_dataset_folder $gpu $num_epochs $is_dev_run $train_for
```
path_to_dataset_folder is the root folder of the dfdc dataset, gpu is the id of the gpu device on which training is to be done, num_epochs is the number of epochs to train the model for, is_dev_run takes a boolean value. If it is true then the model is passed through one batch for training and one batch for testing. train_for takes 2 possible values either clean or noise. If the value of train_for is clean then the frames of the video are not exposed to any noise, and if the value is noise, then the frames of the video are exposed to noise before using for training.  


To train the model on FF++ dataset
```shell
bash eval_all.sh ff $path_to_dataset_folder $num_epochs $is_dev_run $train_for
```

This will create a results folder and store all the y values and y_predicted values as a pickle file, this can then be used by the plot.py script to generate Accuracy scores, AUC scores, ROC curves and confusion matrices.  