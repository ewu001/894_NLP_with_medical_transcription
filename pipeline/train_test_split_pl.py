import pandas as pd
import numpy as np
import math
import os
import argparse

from sklearn.utils import shuffle

def eval_train_generator(path, target, split_ratio, upper_bound=None):
    # upper_bound by default is set to none, which means all data will be sampled
    # this value can be configured to use to make sure equal amount of category will be in the evaluation dataset
    # for example if split ratio is 0.1, then 10% amount of total data for each category will be extracted into evaluation dataset

    dataframe = pd.read_csv(path)
    split_number = math.floor(int(dataframe.shape[0]) * split_ratio)

    true_record = dataframe[dataframe[target]==1]
    false_record = dataframe[dataframe[target]!=1]

    false_record = shuffle(false_record)
    true_record = shuffle(true_record)

    false_eval = false_record.iloc[0:split_number]
    true_eval = true_record.iloc[0:split_number]
    
    if upper_bound:
        false_train = false_record.iloc[split_number+1:upper_bound]
        true_train = true_record.iloc[split_number+1:upper_bound]
    else:
        false_train = false_record.iloc[split_number+1:]
        true_train = true_record.iloc[split_number+1:]

    eval_dataset = pd.concat([true_eval, false_eval])
    eval_dataset = shuffle(eval_dataset)

    train_dataset =  pd.concat([true_train, false_train])
    train_dataset = shuffle(train_dataset)

    print("Created evaluation dataset of shape: ", eval_dataset.shape)
    print("Created training dataset of shape: ", train_dataset.shape)

    eval_path = os.path.join(os.path.dirname(__file__), "../warehouse/store/evaluation_dataset.csv")
    train_path = os.path.join(os.path.dirname(__file__), "../warehouse/store/training_dataset.csv")

    eval_dataset.to_csv(eval_path, encoding='utf-8')
    train_dataset.to_csv(train_path, encoding='utf-8')
    print("Completed")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--category_split_ratio',
        help='The ratio to be used to split the data for evaluation set',
        required=True,
        type=float
    )
    args, _ = parser.parse_known_args()
    hparams = args.__dict__
    split_ratio = hparams['category_split_ratio']
    path = os.path.join(os.path.dirname(__file__), "../warehouse/keyword_target_data.csv")
    eval_train_generator(path, "Target", split_ratio)