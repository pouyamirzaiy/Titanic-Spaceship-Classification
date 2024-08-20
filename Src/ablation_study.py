import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBClassifier

def ablation_study(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    clf = XGBClassifier(n_estimators=352, learning_rate=0.04452171134956829, max_depth=11, min_child_weight=6, subsample=0.9596728088879581, colsample_bytree=0.7321413461163211, gamma=0.8573592725441669, reg_alpha=0.9485599370172857, reg_lambda=0.859332304288036)
    
    results = None
    cv = KFold(n_splits=7)
    features = list(X_train_scaled.columns)
    
    for feature in features:
        X_ablation = X_train_scaled.drop(feature, axis=1)
        
        scores = cross_val_score(clf, X_ablation, y_train, cv=cv)
        mean = scores.mean()
        stdev = scores.std()
        
        clf.fit(X_ablation, y_train)
        y_pred = clf.predict(X_test_scaled.drop(feature, axis=1))
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        row = pd.DataFrame({
            "model": type(clf).__name__,
            "removed_feature": feature,
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
        }, index=[0])
        
        results = pd.concat([results, row], ignore_index=True)
    
    return results

if __name__ == "__main__":
    df = pd.read_csv('/content/engineered_train.csv')
    X = df.drop(columns=["Transported"])
    y = df["Transported"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=8)
    
    results = ablation_study(X_train, X_test, y_train, y_test)
    results.to_csv('/content/ablation_results.csv', index=False)
