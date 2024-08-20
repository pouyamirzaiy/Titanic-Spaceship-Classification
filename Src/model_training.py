import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

def train_and_evaluate(df):
    X = df.drop(columns=["Transported"])
    y = df["Transported"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=8)
    
    mms = MinMaxScaler()
    X_train_scaled = mms.fit_transform(X_train)
    X_test_scaled = mms.transform(X_test)
    
    models = [
        LogisticRegression(max_iter=1000, C=1.0, penalty='l2', solver='lbfgs'),
        DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None),
        RandomForestClassifier(n_estimators=238, max_depth=11, min_samples_split=6, min_samples_leaf=1, max_features='sqrt'),
        GradientBoostingClassifier(n_estimators=311, learning_rate=0.01692577483678813, max_depth=10, min_samples_split=10, min_samples_leaf=7, max_features="log2"),
        AdaBoostClassifier(n_estimators=50, learning_rate=1.0),
        SVC(C=1.0, kernel='rbf', gamma='scale'),
        KNeighborsClassifier(n_neighbors=5),
        GaussianNB(),
        XGBClassifier(n_estimators=352, learning_rate=0.04452171134956829, max_depth=11, min_child_weight=6, subsample=0.9596728088879581, colsample_bytree=0.7321413461163211, gamma=0.8573592725441669, reg_alpha=0.9485599370172857, reg_lambda=0.859332304288036)
    ]
    
    results = None
    cv = KFold(n_splits=7)
    
    for i, model in enumerate(models):
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv)
        mean = scores.mean()
        stdev = scores.std()
        
        model = model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        row = pd.DataFrame({
            "model": type(model).__name__,
            "mean": mean,
            "lower_limit": mean - 2 * stdev,
            "upper_limit": mean + 2 * stdev,
            "MAE": mae,
            "MSE": mse,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "R_positives": conf_matrix[0][0],
            "F_positives": conf_matrix[0][1],
            "R_negatives": conf_matrix[1][1],
            "F_negatives": conf_matrix[1][0]
        }, index=[i])
        
        results = pd.concat([results, row], ignore_index=True)
    
    return results

if __name__ == "__main__":
    df = pd.read_csv('/content/engineered_train.csv')
    results = train_and_evaluate(df)
    results.to_csv('/content/model_results.csv', index=False)
