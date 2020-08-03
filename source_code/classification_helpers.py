import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, \
    KFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, roc_curve

def initial_split(X, y):
    """Splits features and target dataframes in 80/20 ratio.

    Args:
        X: A dataframe only consisting of the features.
        y: A dataframe only consisting of the target.

    Returns:
        X_train_val: A dataframe, containing 80% of the original features data,
            to be used for training and validation.
        X_test: A dataframe, containing 20% of the original features data, to be
            used for testing.
        y_train_val: A dataframe, containing 80% of the original target data, to
            be used for training and validation.
        y_test: A dataframe, containing 20% of the original target data, to be
            used for testing.
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=4444)
    return X_train_val, X_test, y_train_val, y_test

def second_split(X_train_val, y_train_val):
    """Splits features and target dataframes so training set
    is 60% of all data and validation set is 20% of all data.

    Args:
        X_train_val: A dataframe, containing 80% of the original features data,
            to be used for training and validation.
        y_train_val: A dataframe, containing 80% of the original target data, to
            be used for training and validation.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=.25, random_state=4444)
    return X_train, X_val, y_train, y_val

def feature_target_selection(features, target, df):
    """Returns two dataframes, each corresponding to the features and target.

    Args:
        features: A list of features for the model.
        target: The target for the model, passed as a single-element list.

    Returns:
        X: A dataframe only consisting of the features.
        y: A dataframe only consisting ot the target.
    """
    X = df.loc[:, features]
    y = df[target]
    return X, y

def split_and_simple_validate(model, X_train_val, y_train_val, sv_records_df,
                              threshold=0.5, scale=False, xgboost=False):
    """Splits the data into training and validation sets in 75/25 ratio and
    prints scores/intercept/coefficients.

    Args:
        model: Model instance to fit and simple validate.
        threshold: Threshold used to make hard classifications in logistic regression.
        X_train_val: A dataframe, containing 80% of the original features data,
            to be used for training and validation.
        y_train_val: A dataframe, containing 80% of the original target data, to
            be used for training and validation.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=.25, random_state=4444)

    model_name = re.sub(r'\((.*)\)', '', str(model))

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
    model.fit(X_train, y_train)
    if xgboost:
        y_train_pred = np.where(model.predict(X_train,
            ntree_limit=model.best_ntree_limit) > threshold, 1, 0)
        y_val_pred = np.where(model.predict(X_val,
            ntree_limit=model.best_ntree_limit) > threshold, 1, 0)
    else:
        if threshold == 0.5:
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
        else:
            y_train_pred = np.where(model.predict_proba(X_train)[
                                    :, 1] > threshold, 1, 0)
            y_val_pred = np.where(model.predict_proba(X_val)
                                  [:, 1] > threshold, 1, 0)


    hyperparameters = str(model).replace(model_name, '')[1:-1]

    train_f1 = f1_score(y_train, y_train_pred)
    val_f1 = f1_score(y_val, y_val_pred)

    train_precision = precision_score(y_train, y_train_pred)
    val_precision = precision_score(y_val, y_val_pred)

    train_recall = recall_score(y_train, y_train_pred)
    val_recall = recall_score(y_val, y_val_pred)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    train_auc = roc_auc_score(y_train, y_train_pred)
    val_auc = roc_auc_score(y_val, y_val_pred)

    print(f'{"Train F1:": <40} {train_f1: .2f}')
    print(f'{"Val F1:": <40} {val_f1: .2f}')
    print(f'{"Train precision:": <40} {train_precision: .2f}')
    print(f'{"Val precision:": <40} {val_precision: .2f}')
    print(f'{"Train recall:": <40} {train_recall: .2f}')
    print(f'{"Val recall:": <40} {val_recall: .2f}')
    print(f'{"Train accuracy:": <40} {train_accuracy: .2f}')
    print(f'{"Val accuracy:": <40} {val_accuracy: .2f}')
    print(f'{"Train AUC:": <40} {train_auc: .2f}')
    print(f'{"Val AUC:": <40} {val_auc: .2f}')

    if model_name == 'LogisticRegression':
        print('\nFeature coefficients:\n')
        for feature, coef in zip(X_train_val.columns, model.coef_[0]):
            print(f'{feature: <40} {coef: .2f}')
        print(f'\n{"Intercept:": <40} {model.intercept_[0]: .2f}')

    sv_records_df = sv_records_df.append(record_scores(model_name, hyperparameters,
                                                       train_f1, val_f1,
                                                       train_precision, val_precision,
                                                       train_recall, val_recall,
                                                       train_accuracy, val_accuracy,
                                                       train_auc, val_auc),
                                         ignore_index=True)

    return model, sv_records_df

def cv(model, X_train_val, y_train_val, cv_records_df):
    """Performs 5-fold cross validation and prints training and test scores.
    Also adds scores to cv_records, which is a list of dicts.

    Args:
        model: Model instance to perform cross validation on.
        X_train_val: A dataframe, containing 80% of the original features data,
            to be used for training and validation.
        y_train_val: A dataframe, containing 80% of the original target data, to
            be used for training and validation.
        cv_records_df: A dataframe to record cross validation scores.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=4444)
    scores = cross_validate(model, X_train_val, y_train_val,
                            cv=kf, scoring=['f1', 'precision',
                                            'recall', 'accuracy',
                                            'roc_auc'],
                            return_train_score=True)

    model_name = re.sub(r'\((.*)\)', '', str(model))
    hyperparameters = str(model).replace(model_name, '')[1:-1]
    mean_train_f1 = np.mean(scores['train_f1'])
    mean_val_f1 = np.mean(scores['test_f1'])
    mean_train_precision = np.mean(scores['train_precision'])
    mean_val_precision = np.mean(scores['test_precision'])
    mean_train_recall = np.mean(scores['train_recall'])
    mean_val_recall = np.mean(scores['test_recall'])
    mean_train_accuracy = np.mean(scores['train_accuracy'])
    mean_val_accuracy = np.mean(scores['test_accuracy'])
    mean_train_auc = np.mean(scores['train_roc_auc'])
    mean_val_auc = np.mean(scores['test_roc_auc'])

    print(f'Model name: {model_name}')
    print(f'Hyperparameters: {hyperparameters}\n')

    print(f'{"Mean train F1:": <25} {mean_train_f1: .3f}')
    print(f'{"Mean val F1:": <25} {mean_val_f1: .3f}')
    print(f'{"Mean train precision:": <25} {mean_train_precision: .3f}')
    print(f'{"Mean val precision:": <25} {mean_val_precision: .3f}')
    print(f'{"Mean train recall:": <25} {mean_train_recall: .3f}')
    print(f'{"Mean val recall:": <25} {mean_val_recall: .3f}')
    print(f'{"Mean train accuracy:": <25} {mean_train_accuracy: .3f}')
    print(f'{"Mean val accuracy:": <25} {mean_val_accuracy: .3f}')
    print(f'{"Mean train AUC:": <25} {mean_train_auc: .3f}')
    print(f'{"Mean val AUC:": <25} {mean_val_auc: .3f}')

    cv_records_df = cv_records_df.append(record_scores(model_name, hyperparameters,
                                                       mean_train_f1, mean_val_f1,
                                                       mean_train_precision, mean_val_precision,
                                                       mean_train_recall, mean_val_recall,
                                                       mean_train_accuracy, mean_val_accuracy,
                                                       mean_train_auc, mean_val_auc),
                                         ignore_index=True)
    return cv_records_df

def record_scores(model_name, hyperparameters,
                  mean_train_f1, mean_val_f1,
                  mean_train_precision, mean_val_precision,
                  mean_train_recall, mean_val_recall,
                  mean_train_accuracy, mean_val_accuracy,
                  mean_train_auc, mean_val_auc):
    """Records scores with other record-keeping information
    in a dict.

    Args:
        model_name: The model's name.
        hyperparameters: The model's hyperparameters.
        mean_train_f1: The mean cross validation training F1 score.
        mean_val_f1: The mean cross validation validation F1 score.
        mean_train_precision: The mean cross validation training precision score.
        mean_val_precision: The mean cross validation validation precision score.
        mean_train_recall: The mean cross validation training recall score.
        mean_val_recall: The mean cross validation validation recall score.
        mean_train_accuracy: The mean cross validation training accuracy score.
        mean_val_accuracy: The mean cross validation validation accuracy score.
        mean_train_auc: The mean cross validation training AUC score.
        mean_val_auc: The mean cross validation validation AUC score.

    Returns:
        scores_dict: A dict of cross valiation scores with other record-keeping
            information.
    """
    scores_dict = {}
    desc = input("iteration_desc: ")
    feature_eng = input("feature_engineering: ")

    scores_dict['model'] = model_name
    scores_dict['iteration_desc'] = desc
    scores_dict['feature_engineering'] = feature_eng
    scores_dict['hyperparameter_tuning'] = hyperparameters

    scores_dict['mean_train_f1'] = float(f'{mean_train_f1: .3f}')
    scores_dict['mean_val_f1'] = float(f'{mean_val_f1: .3f}')
    scores_dict['mean_train_precision'] = float(f'{mean_train_precision: .3f}')
    scores_dict['mean_val_precision'] = float(f'{mean_val_precision: .3f}')
    scores_dict['mean_train_recall'] = float(f'{mean_train_recall: .3f}')
    scores_dict['mean_val_recall'] = float(f'{mean_val_recall: .3f}')
    scores_dict['mean_train_accuracy'] = float(f'{mean_train_accuracy: .3f}')
    scores_dict['mean_val_accuracy'] = float(f'{mean_val_accuracy: .3f}')
    scores_dict['mean_train_AUC'] = float(f'{mean_train_auc: .3f}')
    scores_dict['mean_val_AUC'] = float(f'{mean_val_auc: .3f}')

    return scores_dict

def plot_ROC(model, X_train, X_val, y_train, y_val):
    model.fit(X_train, y_train)
    fpr, tpr, thresholds = roc_curve(y_val, model.predict_proba(X_val)[:, 1])
    model_name = re.sub(r'\((.*)\)', '', str(model))
    plt.plot(fpr, tpr, lw=2, label=model_name)
    plt.plot([0, 1], [0, 1], c='violet', ls='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False positive rate (FPR)')
    plt.ylabel('True positive rate (TPR)')
    plt.title('ROC curves')
    plt.legend()
    return fpr, tpr, thresholds

def get_best_threshold_from_roc_curve(fpr, tpr, thresholds):
    """Returns best combined tpr/fpr score and the threshold associated with that score.

    Args:
        fpr: False positive rates (fp/(fp+tn)).
        tpr: True positive rates (tp/tp+fn).

    Returns:
        best_score: Highest combined tpr/fpr score (tpr - fpr).
        best_threshold: Threshold that corresponds to best_score.
        """
    combined_score = tpr - fpr
    best_threshold = thresholds[np.argmax(combined_score)]
    best_score = np.amax(combined_score)
    return best_score, best_threshold

def pairplot_features(df):
    """Displays pairplot for given set of features with distributions
    differentiated by the target class is_not_on_time.

    Args:
        df: Dataframe with desired features and target.
    """
    sample = df.sample(10000, random_state=4444)
    sns.pairplot(sample, hue='is_not_on_time', plot_kws=dict(alpha=0.3))
