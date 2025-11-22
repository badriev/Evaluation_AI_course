#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("=" * 80)
print("1. ЗАГРУЗКА И АНАЛИЗ ДАТАСЕТА DIAMONDS")
print("=" * 80)

print("📊 Загружаем датасет Diamonds...")
diamonds = sns.load_dataset('diamonds')

print("\n📋 Первые 5 строк датасета:")
print(diamonds.head())

print(f"\n📈 Размер датасета: {diamonds.shape}")
print(f"📊 Количество признаков: {diamonds.shape[1]}")
print(f"🎯 Количество образцов: {diamonds.shape[0]}")

print("\n🔍 Информация о данных:")
diamonds.info()

print("\n📊 Статистика численных признаков:")
print(diamonds.describe())

plt.figure(figsize=(10, 6))
plt.hist(diamonds['price'], bins=50, edgecolor='black', alpha=0.7)
plt.title('Распределение цен алмазов')
plt.xlabel('Цена ($)')
plt.ylabel('Количество алмазов')
plt.grid(True, alpha=0.3)
plt.show()

print("💡 Наблюдения:")
print("- Цены варьируются от 326 до 18823 долларов")
print("- Большинство алмазов стоят менее 5000 долларов")
print("- Распределение смещено вправо (длинный хвост дорогих алмазов)")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

cut_counts = diamonds['cut'].value_counts()
axes[0].bar(cut_counts.index, cut_counts.values, color='skyblue', edgecolor='black')
axes[0].set_title('Распределение качества огранки')
axes[0].set_xlabel('Качество огранки')
axes[0].set_ylabel('Количество')
axes[0].tick_params(axis='x', rotation=45)

color_counts = diamonds['color'].value_counts().sort_index()
axes[1].bar(color_counts.index, color_counts.values, color='lightgreen', edgecolor='black')
axes[1].set_title('Распределение цвета алмазов')
axes[1].set_xlabel('Цвет (D - лучший, J - худший)')
axes[1].set_ylabel('Количество')

clarity_counts = diamonds['clarity'].value_counts()
axes[2].bar(clarity_counts.index, clarity_counts.values, color='lightcoral', edgecolor='black')
axes[2].set_title('Распределение чистоты алмазов')
axes[2].set_xlabel('Чистота')
axes[2].set_ylabel('Количество')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("💡 Наблюдения:")
print("- Большинство алмазов имеют хорошее или очень хорошее качество огранки")
print("- Цвет G является самым распространенным")
print("- Чистота VS2 и SI1 наиболее распространены")

numeric_cols = ['carat', 'depth', 'table', 'x', 'y', 'z', 'price']
correlation_matrix = diamonds[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('Корреляционная матрица численных признаков')
plt.tight_layout()
plt.show()

print("💡 Важные наблюдения:")
print("- Carat имеет сильную положительную корреляцию с ценой (0.92)")
print("- Размеры x, y, z также сильно коррелируют с carat и ценой")
print("- Depth и table имеют слабую корреляцию с ценой")
print("- Высокая корреляция между размерами может указывать на мультиколлинеарность")

print("\n" + "=" * 80)
print("2. ПРЕДОБРАБОТКА ДАННЫХ")
print("=" * 80)

print("🔄 Кодируем категориальные признаки...")

data = diamonds.copy()

le_cut = LabelEncoder()
le_color = LabelEncoder()
le_clarity = LabelEncoder()

data['cut_encoded'] = le_cut.fit_transform(data['cut'])
data['color_encoded'] = le_color.fit_transform(data['color'])
data['clarity_encoded'] = le_clarity.fit_transform(data['clarity'])

print("\n📋 Кодировка категорий:")
cut_dict = dict(zip(le_cut.classes_, range(len(le_cut.classes_))))
color_dict = dict(zip(le_color.classes_, range(len(le_color.classes_))))
clarity_dict = dict(zip(le_clarity.classes_, range(len(le_clarity.classes_))))
print(f"Cut: {cut_dict}")
print(f"Color: {color_dict}")
print(f"Clarity: {clarity_dict}")

print("\n📊 Первые 5 строк после кодирования:")
print(data[['cut', 'cut_encoded', 'color', 'color_encoded', 'clarity', 'clarity_encoded', 'price']].head())

feature_cols = ['carat', 'depth', 'table', 'x', 'y', 'z', 'cut_encoded', 'color_encoded', 'clarity_encoded']
X = data[feature_cols]

y = data['price']

print(f"\n🎯 Признаки (X): {list(X.columns)}")
print(f"📊 Размер X: {X.shape}")
print(f"💰 Целевая переменная (y): price")
print(f"📊 Размер y: {y.shape}")

print("\n🔄 Нормализуем численные признаки...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\n📊 Статистика после нормализации:")
print(X_scaled.describe().round(3))

print("\n💡 Почему нормализация важна:")
print("- Carat варьируется от 0.2 до 5.01, а depth от 43 до 79")
print("- Без нормализации модель будет считать carat более важным просто из-за масштаба")
print("- После нормализации все признаки имеют среднее ~0 и std ~1")

print("\n🎯 Разделяем данные на обучающую и тестовую выборки...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"📚 Обучающая выборка: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"🧪 Тестовая выборка: X_test={X_test.shape}, y_test={y_test.shape}")
print(f"📊 Размер тестовой выборки: {len(X_test)} образцов ({100*len(X_test)/(len(X_train)+len(X_test)):.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].hist(y_train, bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0].set_title('Распределение цен в обучающей выборке')
axes[0].set_xlabel('Цена ($)')
axes[0].set_ylabel('Количество')
axes[0].grid(True, alpha=0.3)

axes[1].hist(y_test, bins=50, alpha=0.7, color='red', edgecolor='black')
axes[1].set_title('Распределение цен в тестовой выборке')
axes[1].set_xlabel('Цена ($)')
axes[1].set_ylabel('Количество')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("💡 Наблюдения:")
print("- Распределения похожи, что хорошо для тестирования")
print(f"- Средняя цена в обучении: ${y_train.mean():.0f}")
print(f"- Средняя цена в тесте: ${y_test.mean():.0f}")

print("\n" + "=" * 80)
print("3. СОЗДАНИЕ НЕЙРОННОЙ СЕТИ ДЛЯ РЕГРЕССИИ")
print("=" * 80)

class DiamondsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = DiamondsDataset(X_train, y_train)
test_dataset = DiamondsDataset(X_test, y_test)

print(f"📚 Создан обучающий датасет: {len(train_dataset)} образцов")
print(f"🧪 Создан тестовый датасет: {len(test_dataset)} образцов")

sample_X, sample_y = train_dataset[0]
print(f"\n🔍 Пример данных:")
print(f"Признаки: {sample_X}")
print(f"Цена: ${sample_y.item():.0f}")

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"\n📦 DataLoader создан с размером батча: {batch_size}")
print(f"🔄 Количество батчей в обучении: {len(train_loader)}")
print(f"🔄 Количество батчей в тесте: {len(test_loader)}")

batch_X, batch_y = next(iter(train_loader))
print(f"\n🔍 Пример батча:")
print(f"Размер батча X: {batch_X.shape}")
print(f"Размер батча y: {batch_y.shape}")
print(f"Признаки в батче: {X_train.shape[1]}, Цели: 1 (регрессия)")

print("\n💡 Преимущества батчевой обработки:")
print("- Экономит память (не загружаем все данные сразу)")
print("- Ускоряет обучение (параллельные вычисления)")
print("- Помогает избегать локальных минимумов")

class DiamondPricePredictor(nn.Module):
    def __init__(self, input_size):
        super(DiamondPricePredictor, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)

input_size = X_train.shape[1]
model = DiamondPricePredictor(input_size)

print(f"🧠 Создана нейронная сеть:")
print(f"Вход: {input_size} признаков")
print(f"Архитектура: {input_size} -> 128 -> 64 -> 32 -> 1")
print(f"Активации: ReLU на скрытых слоях")
print(f"Dropout: 20% на первых двух слоях")

with torch.no_grad():
    sample_output = model(batch_X[:5])
    print(f"\n🔍 Пример предсказаний:")
    print(f"Истинные цены: {batch_y[:5].flatten()}")
    print(f"Предсказания: {sample_output.flatten()}")
    print("(Предсказания случайные, так как модель не обучена)")

print("\n📉 Функция потерь: MSE (Mean Squared Error)")
print(f"🎯 MSE = (1/n) * Σ(predicted - true)²")
print(f"⚡ Оптимизатор: Adam с learning rate = {0.001}")

criterion = nn.MSELoss()

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
optimizer.zero_grad()

outputs = model(batch_X)
loss = criterion(outputs, batch_y)

print(f"\n🔍 Пример расчета потерь:")
print(f"Потери на случайной модели: {loss.item():.2f}")
print(f"Это означает среднюю квадратичную ошибку: ${loss.item()**0.5:.0f} (RMSE)")

print("\n💡 Почему MSE для регрессии:")
print("- Штрафует большие ошибки сильнее (квадрат)")
print("- Дифференцируема (можно вычислять градиенты)")
print("- Интерпретируема (в единицах цены в квадрате)")

print("\n" + "=" * 80)
print("4. ОБУЧЕНИЕ МОДЕЛИ")
print("=" * 80)

num_epochs = 50
train_losses = []
test_losses = []

print(f"🚀 Начинаем обучение на {num_epochs} эпохах...")
print("Это может занять несколько минут...\n")

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        outputs = model(batch_X)

        loss = criterion(outputs, batch_y)

        loss.backward()

        optimizer.step()

        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    epoch_test_loss = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            epoch_test_loss += loss.item()

    avg_test_loss = epoch_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Эпоха [{epoch+1}/{num_epochs}]:")
        print(f"  📚 Train Loss: {avg_train_loss:.4f} (${avg_train_loss**0.5:.0f} RMSE)")
        print(f"  🧪 Test Loss: {avg_test_loss:.4f} (${avg_test_loss**0.5:.0f} RMSE)")
        print()

print("✅ Обучение завершено!")
print(f"\n📊 Финальные результаты:")
print(f"Train Loss: {train_losses[-1]:.4f}")
print(f"Test Loss: {test_losses[-1]:.4f}")
print(f"Train RMSE: ${train_losses[-1]**0.5:.0f}")
print(f"Test RMSE: ${test_losses[-1]**0.5:.0f}")

plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Обучение', linewidth=2)
plt.plot(test_losses, label='Тест', linewidth=2)
plt.xlabel('Эпоха')
plt.ylabel('MSE Loss')
plt.title('Процесс обучения нейронной сети')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("💡 Анализ графика обучения:")
print("- Обе кривые должны уменьшаться - модель учится")
print("- Train loss обычно ниже test loss")
print("- Если test loss перестает уменьшаться, может быть переобучение")
print("- Хороший результат: обе кривые стабилизируются на низком уровне")

if train_losses[-1] < test_losses[-1] * 0.7:
    print("\n⚠️  Возможное переобучение: модель слишком хорошо запомнила обучающие данные")
else:
    print("\n✅ Модель обучается хорошо, без явного переобучения")

print("\n" + "=" * 80)
print("5. ОЦЕНКА КАЧЕСТВА МОДЕЛИ")
print("=" * 80)

model.eval()
all_predictions = []
all_true_values = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        all_predictions.extend(outputs.numpy().flatten())
        all_true_values.extend(batch_y.numpy().flatten())

y_pred = np.array(all_predictions)
y_true = np.array(all_true_values)

print(f"🔍 Получено предсказаний: {len(y_pred)}")
print(f"🔍 Истинных значений: {len(y_true)}")

print("\n📋 Примеры предсказаний:")
for i in range(10):
    pred = y_pred[i]
    true = y_true[i]
    error = abs(pred - true)
    print(f"Образец {i+1}: Предсказано ${pred:.0f}, Истинная ${true:.0f}, Ошибка ${error:.0f}")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

mae = mean_absolute_error(y_true, y_pred)

r2 = r2_score(y_true, y_pred)

print("📊 МЕТРИКИ КАЧЕСТВА МОДЕЛИ:")
print("=" * 50)
print(f"📏 MSE (Mean Squared Error): {mse:.2f}")
print(f"📐 RMSE (Root MSE): ${rmse:.2f}")
print(f"📊 MAE (Mean Absolute Error): ${mae:.2f}")
print(f"🎯 R² Score: {r2:.4f} ({r2*100:.1f}%)")
print("=" * 50)

print("\n💡 ИНТЕРПРЕТАЦИЯ МЕТРИК:")
print(f"- RMSE = ${rmse:.0f}: средняя ошибка предсказания цены")
print(f"- MAE = ${mae:.0f}: средняя абсолютная ошибка")
print(f"- R² = {r2:.1%}: модель объясняет {r2*100:.1f}% вариации цен")
print(f"- R² = 1.0: идеальная модель")
print(f"- R² = 0.0: модель не лучше среднего")
print(f"- R² < 0.0: модель хуже среднего")

mean_price = y_true.mean()
naive_mae = mean_absolute_error(y_true, [mean_price] * len(y_true))
print(f"\n🔍 Сравнение с наивной моделью (всегда предсказывает среднюю цену):")
print(f"Наша модель MAE: ${mae:.0f}")
print(f"Наивная модель MAE: ${naive_mae:.0f}")
print(f"Улучшение: {((naive_mae - mae) / naive_mae * 100):.1f}%")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0,0].scatter(y_true, y_pred, alpha=0.6, color='blue', s=30)
axes[0,0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
               'r--', linewidth=2, label='Идеальная линия')
axes[0,0].set_xlabel('Истинная цена ($)')
axes[0,0].set_ylabel('Предсказанная цена ($)')
axes[0,0].set_title('Предсказанные vs Истинные цены')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

errors = y_pred - y_true
axes[0,1].hist(errors, bins=50, alpha=0.7, color='red', edgecolor='black')
axes[0,1].axvline(x=0, color='black', linestyle='--', linewidth=2, label='Нет ошибки')
axes[0,1].set_xlabel('Ошибка предсказания ($)')
axes[0,1].set_ylabel('Количество образцов')
axes[0,1].set_title('Распределение ошибок')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

axes[1,0].scatter(y_true, errors, alpha=0.6, color='green', s=30)
axes[1,0].axhline(y=0, color='black', linestyle='--', linewidth=2)
axes[1,0].set_xlabel('Истинная цена ($)')
axes[1,0].set_ylabel('Ошибка предсказания ($)')
axes[1,0].set_title('Ошибки vs Истинные значения')
axes[1,0].grid(True, alpha=0.3)

from scipy import stats
stats.probplot(errors, dist="norm", plot=axes[1,1])
axes[1,1].set_title('Q-Q plot ошибок')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("АНАЛИЗ РЕЗУЛЬТАТОВ НЕЙРОННОЙ СЕТИ")
print("=================================")

try:
    r2
except NameError:
    print("❌ Ошибка: Переменная 'r2' не определена. Запустите предыдущую ячейку с метриками.")
    r2 = 0.0

try:
    mae
except NameError:
    print("❌ Ошибка: Переменная 'mae' не определена. Запустите предыдущую ячейку с метриками.")
    mae = 0.0

try:
    rmse
except NameError:
    print("❌ Ошибка: Переменная 'rmse' не определена. Запустите предыдущую ячейку с метриками.")
    rmse = 0.0

try:
    y_true
except NameError:
    print("❌ Ошибка: Переменная 'y_true' не определена. Запустите ячейку с предсказаниями.")
    y_true = np.array([0])

print(f"📊 R² Score: {r2:.1%} - модель объясняет {r2*100:.1f}% вариации цен")
print(f"💰 Средняя ошибка: ${mae:.0f} на алмаз")
print(f"📏 RMSE: ${rmse:.0f} - типичная ошибка предсказания")

price_range = y_true.max() - y_true.min()
error_percentage = (mae / price_range) * 100 if price_range > 0 else 0
print(f"\n📈 Диапазон цен в данных: ${y_true.min():.0f} - ${y_true.max():.0f} (${price_range:.0f})")
print(f"📊 Ошибка относительно диапазона: {error_percentage:.1f}%")

if r2 > 0.9:
    quality = "ОТЛИЧНОЕ"
    comment = "Модель очень точно предсказывает цены алмазов"
elif r2 > 0.8:
    quality = "ХОРОШЕЕ"
    comment = "Модель достаточно точно предсказывает цены"
elif r2 > 0.7:
    quality = "УДОВЛЕТВОРИТЕЛЬНОЕ"
    comment = "Модель работает приемлемо, но есть пространство для улучшения"
else:
    quality = "ПЛОХОЕ"
    comment = "Модель нуждается в значительных улучшениях"

print(f"\n🎯 КАЧЕСТВО МОДЕЛИ: {quality}")
print(f"💡 {comment}")

print("\n💡 ВОЗМОЖНЫЕ УЛУЧШЕНИЯ МОДЕЛИ:")
print("1. Увеличить количество эпох обучения")
print("2. Изменить архитектуру сети (больше/меньше слоев)")
print("3. Попробовать другие оптимизаторы (SGD, RMSprop)")
print("4. Добавить регуляризацию (L1/L2)")
print("5. Использовать кросс-валидацию")
print("6. Обработать выбросы в данных")
print("7. Попробовать другие методы кодирования категориальных признаков")

print("\n✅ ДОМАШНЕЕ ЗАДАНИЕ ВЫПОЛНЕНО!")
print("Мы успешно применили глубокое обучение для предсказания цен алмазов.")
print("Модель показывает ", end="")
if r2 > 0.8:
    print("хорошие результаты")
else:
    print("удовлетворительные результаты")
print("и может быть использована для оценки стоимости алмазов.")

import numpy as np

if __name__ == "__main__":
    print("\n🚀 Скрипт выполнен успешно!")
