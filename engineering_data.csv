import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_data(filepath):
    """Загрузка данных из CSV-файла."""
    try:
        data = pd.read_csv(filepath)
        print("Данные успешно загружены.")
        return data
    except FileNotFoundError:
        print(f"Файл {filepath} не найден.")
        return None

def clean_data(data):
    """Очистка данных: обработка пропущенных значений и удаление выбросов."""
    print("Начинается очистка данных...")

    # Обработка пропущенных значений
    missing_before = data.isnull().sum()
    print("Пропущенные значения до очистки:")
    print(missing_before)

    data = data.dropna()  # Удаляем строки с пропущенными значениями
    # Альтернативно можно заполнить пропущенные значения, например:
    # data.fillna(data.mean(), inplace=True)

    missing_after = data.isnull().sum()
    print("Пропущенные значения после очистки:")
    print(missing_after)

    # Удаление выбросов с использованием Z-оценки
    from scipy import stats
    z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
    filtered_entries = (z_scores < 3).all(axis=1)
    data = data[filtered_entries]
    print("Выбросы удалены.")

    print("Очистка данных завершена.")
    return data

def visualize_data(data):
    """Визуализация данных."""
    print("Начинается визуализация данных...")
    sns.set(style="whitegrid")

    # Гистограммы для числовых переменных
    data.hist(bins=30, figsize=(15,10))
    plt.tight_layout()
    plt.show()

    # Корреляционная матрица
    plt.figure(figsize=(10,8))
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Корреляционная матрица")
    plt.show()

    # Рассеяние для пары переменных
    sns.pairplot(data)
    plt.show()

    print("Визуализация данных завершена.")

def statistical_analysis(data):
    """Выполнение описательного статистического анализа."""
    print("Описательные статистики:")
    print(data.describe())

def machine_learning_model(data):
    """Пример простой регрессионной модели."""
    print("Начинается построение модели машинного обучения...")

    # Предположим, что мы хотим предсказать напряжение (Stress)
    X = data[['Dimension1', 'Dimension2', 'Dimension3', 'Temperature']]
    y = data['Stress']

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание и обучение модели
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Модель обучена.")

    # Предсказания
    y_pred = model.predict(X_test)

    # Оценка модели
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
    print(f"Коэффициент детерминации (R²): {r2:.2f}")

    # Визуализация предсказаний
    plt.scatter(y_test, y_pred)
    plt.xlabel("Фактические значения")
    plt.ylabel("Предсказанные значения")
    plt.title("Фактические vs Предсказанные значения")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.show()

    print("Моделирование завершено.")

def main():
    filepath = 'engineering_data.csv'  # Укажите путь к вашему файлу
    data = load_data(filepath)
    if data is not None:
        data = clean_data(data)
        visualize_data(data)
        statistical_analysis(data)
        machine_learning_model(data)

if __name__ == "__main__":
    main()