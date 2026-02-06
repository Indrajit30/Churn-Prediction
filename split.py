import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data, target, test_size=0.2, random_state=42):
    x_train, x_, y_train, y_ = train_test_split(data, target, test_size=test_size, random_state=random_state)
    x_val, x_test, y_val, y_test = train_test_split(x_, y_, test_size=0.5, random_state=random_state)
    del x_, y_
    return x_train, x_val, x_test, y_train, y_val, y_test