import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def prepare_data(file_name: str):
    """
    Подготовка данных для обучения и предсказания.
    Данные представляют собой датафрейм с признаками машин.

    Param:
    file_name: str

    Return:
    x: pd.DataFrame
    y: pd.Series

    """
    df = pd.read_csv(file_name, sep=';')
    y = df['Price']
    x = df.drop(['Price'], axis=1)[['Mileage', 'Cylinders', 'Airbags', 'Prod year']]
    return x, y


def fitting(x_train: pd.DataFrame, y_train: pd.Series, model_name: str, id_model: int, model_params: dict):
    """
    На вход принимает признаки и таргет, название модели и id, которую нужно обучить и параменты этой модели.

    Param:
    x_train: pd.DataFrame
    y_train: pd.Series
    model_name: str
    id_model: int
    model_params: dict

    """
    models = {'RandomForestRegressor': RandomForestRegressor(), 'LinearRegression': LinearRegression()}
    for param in model_params:
        if param not in models[model_name].get_params().keys():
            return 'error', param
    model = models[model_name].set_params(**model_params)
    model.fit(x_train, y_train)
    with open('fitted_models/' + str(id_model) + '.pkl', 'wb') as file:
        pickle.dump(model, file)
    return 'ok', 1


def prediction(id_model: int, x_test: pd.DataFrame):
    """
    На вход принимает id модели и тестовую выборку признаков

    Param:
    x_test: pd.DataFrame
    id_model: int

    Return:
    y_pred: pd.Series

    """

    with open('fitted_models/' + str(id_model) + '.pkl', 'rb') as file:
        fitted_model = pickle.load(file)
    y_pred = fitted_model.predict(x_test)
    return y_pred