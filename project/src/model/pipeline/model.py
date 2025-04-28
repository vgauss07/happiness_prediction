import pandas as pd
import pickle as pk
from loguru import logger

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from config import model_settings
from model.pipeline.preparation import prepare_data


def build_model():
    # train and save the model we need
    logger.info("Starting up Model Building Pipeline")
    # load processed data
    df = prepare_data()
    feature_names = ['gdp',
                     'support',
                     'healthy',
                     'freedom',
                     'corruption']
    logger.info("Building Model.")
    # 2. identify X and y
    X, y = _get_X_y(df, col_X=feature_names)
    # 3. Split Dataset
    X_train, X_test, y_train, y_test = _split_train_test(X, y)
    # 4. train the model
    rF = _train_model(X_train, y_train)
    # 5. Evaluate Model
    _evaluate_model(rF, X_test, y_test)
    # 6. Save Model
    _save_model(rF)


def _get_X_y(dataframe: pd.DataFrame,
             col_X: list[str],
             col_y: str = 'score'):
    logger.info(f'Defining X and Y variables.'
                f'\nX vars: {col_X}\ny var: {col_y}')
    return dataframe[col_X], dataframe[col_y]


def _split_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=32)
    return X_train, X_test, y_train, y_test


def _train_model(X_train, y_train):
    grid_space = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9, 12]}

    grid = GridSearchCV(RandomForestRegressor(),
                        param_grid=grid_space,
                        cv=5,
                        scoring='r2')
    logger.debug(f'grid_space = {grid_space}')

    model_grid = grid.fit(X_train, y_train)
    return model_grid.best_estimator_


def _evaluate_model(model, X_test, y_test):
    logger.info(f'Evaluating Model Performance Score = '
                f'{model.score(X_test, y_test)}')
    return model.score(X_test, y_test)


def _save_model(model):
    logger.info(f'Saving the model:'
                f'{model_settings.model_path}/'
                f'{model_settings.model_name}')
    pk.dump(model,
            open(f'{model_settings.model_path}/'
                 f'{model_settings.model_name}', 'wb'))


# test
build_model()
