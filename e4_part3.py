import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Загрузка данных из CSV-файла
data = pd.read_csv("train.csv")

# Определяем признаки (X) и целевую переменную (y)
X = data[['d1', 'd2']]  # Признаки 'd1' и 'd2'
y = data['d3']           # Целевая переменная 'd3'

# Разделение данных на обучающий и тестовый наборы (например, 80% обучающего, 20% тестового)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели машинного обучения (например, линейная регрессия)
model = LinearRegression()
model.fit(X_train, y_train)

# Оценка модели на тестовых данных
score = model.score(X_test, y_test)
print("Score:", score)
