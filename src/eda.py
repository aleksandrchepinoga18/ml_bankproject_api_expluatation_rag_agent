import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(df: pd.DataFrame, output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)

    print("\n[INFO] Общая информация:")
    print(df.info())

    print("Описательная статистика:")
    print(df.describe())

    print(" Пропуски по столбцам:")
    print(df.isnull().sum())

    # Boxplots
    numeric_cols = df.select_dtypes(include='number').columns.drop('target', errors='ignore')
    print(f"Число числовых признаков без целевой: {len(numeric_cols)}")

    for i in range(0, len(numeric_cols), 10):
        cols_to_plot = numeric_cols[i:i+10]
        df[cols_to_plot].plot(kind='box', subplots=True, layout=(2, 5), figsize=(20, 8))
        plt.suptitle(f'Boxplots: Признаки {i+1} — {i+len(cols_to_plot)}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{output_dir}/boxplot_{i}.png")
        plt.close()
