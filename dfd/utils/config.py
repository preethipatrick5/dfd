import os.path


def get_dfdc_metadata_config(root):
    dfdc_config = {
        "train_file": os.path.join(root, "train.data"),
        "val_file": os.path.join(root, "val.data"),
        "test_file": os.path.join(root, "test.data")
    }
    return dfdc_config


def get_ff_metadata_config(root):
    ff_config = {
        "train_file": os.path.join(root, "train.csv"),
        "val_file": os.path.join(root, "val.csv"),
        "test_file": os.path.join(root, "test.csv")
    }
    return ff_config
