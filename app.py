import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def clean_data(data):
    data = data.dropna()
    from scipy import stats
    z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
    filtered_entries = (z_scores < 3).all(axis=1)
    data = data[filtered_entries]
    return data

def main():
    st.title("Анализ данных в области машиностроения")

    uploaded_file = st.file_uploader("Загрузите CSV-файл", type="csv")
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("**Исходные данные:**")
        st.dataframe(data.head())

        data = clean_data(data)
        st.write("**Данные после очистки:**")
        st.dataframe(data.head())

        st.write("**Описательные статистики:**")
        st.write(data.describe())

        st.write("**Корреляционная матрица:**")
        corr = data.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Машинное обучение
        st.write("**Машинное обучение: Линейная регрессия**")
        if 'Stress' in data.columns:
            X = data[['Dimension1', 'Dimension2', 'Dimension3', 'Temperature']]
            y = data['Stress']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
            st.write(f"Коэффициент детерминации (R²): {r2:.2f}")

            fig2, ax2 = plt.subplots()
            ax2.scatter(y_test, y_pred)
            ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            ax2.set_xlabel("Фактические значения")
            ax2.set_ylabel("Предсказанные значения")
            ax2.set_title("Фактические vs Предсказанные значения")
            st.pyplot(fig2)
        else:
            st.write("Столбец 'Stress' не найден в данных.")

if __name__ == "__main__":
    main()