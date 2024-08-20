import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_visualizations(df):
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    ax = ax.flatten()
    
    numerical_variables = df.select_dtypes(include=["float64"])
    numerical_variables.replace([float('inf'), float('-inf')], np.nan, inplace=True)
    
    for i, column in enumerate(numerical_variables.columns):
        sns.kdeplot(numerical_variables[column], ax=ax[i], color="#334670", fill=True)
        sns.rugplot(numerical_variables[column], ax=ax[i], color="#458284", height=0.05, expand_margins=True)
        ax[i].set_title(f"{column.capitalize()}", fontweight="bold")
        ax[i].set_xlabel("")
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 8))
    mask = np.triu(np.ones_like(numerical_variables.corr(), dtype=bool))
    sns.heatmap(numerical_variables.corr(), cmap="crest", annot=True, mask=mask)
    plt.title("Correlation Map", fontsize=14)
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv('/content/cleaned_train.csv')
    create_visualizations(df)
