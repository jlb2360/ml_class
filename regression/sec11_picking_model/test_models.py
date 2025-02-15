import numpy as np
import pandas as pd
import sklearn as skl


def multi_linear(X, y):
    
    X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(X, y, test_size = 0.2, random_state = 0)

    ml_regres = skl.linear_model.LinearRegression()
    ml_regres.fit(X_train, y_train)

    y_pred = ml_regres.predict(X_test)

    score = skl.metrics.r2_score(y_test, y_pred)

    return score

def poly(X, y):
    X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(X, y, test_size = 0.2, random_state = 0)
    poly_reg = skl.preprocessing.PolynomialFeatures(degree=4)
    X_poly = poly_reg.fit_transform(X_train)
    regressor = skl.linear_model.LinearRegression()
    regressor.fit(X_poly, y_train)

    y_pred = regressor.predict(poly_reg.transform(X_test))

    score = skl.metrics.r2_score(y_test, y_pred)
    
    return score

def decision_tree(X, y):
    X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(X, y, test_size = 0.2, random_state = 0)

    clf = skl.tree.DecisionTreeRegressor(random_state=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    score = skl.metrics.r2_score(y_test, y_pred)

    return score


def random_forest(X, y):
    X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(X, y, test_size = 0.2, random_state = 0)

    clf = skl.ensemble.RandomForestRegressor(random_state=0, n_estimators=10)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    score = skl.metrics.r2_score(y_test, y_pred)

    return score




if __name__ == "__main__":
    dataset = pd.read_csv("Data.csv")
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values

    multi_linear_score = multi_linear(X, y)
    poly_score = poly(X, y)
    dec_tree_score = decision_tree(X, y)
    forest_score = random_forest(X, y)
    
    print(f'R2 score for multivariable linear regression: {multi_linear_score}')
    print(f'R2 score for polynomial regression: {poly_score}')
    print(f'R2 score for decision tree: {dec_tree_score}')
    print(f'R2 score for forest_score: {forest_score}')


    
