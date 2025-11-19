import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Инициализация session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'scaler_x' not in st.session_state:
    st.session_state.scaler_x = None
if 'scaler_y' not in st.session_state:
    st.session_state.scaler_y = None
if 'history' not in st.session_state:
    st.session_state.history = None

def create_neural_network(input_dim, hidden_layers, hidden_units, dropout_rate, learning_rate):
    """Создание нейронной сети"""
    model = Sequential()
    
    # Входной слой
    model.add(Dense(hidden_units, activation='relu', input_dim=input_dim))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    # Скрытые слои
    for _ in range(hidden_layers - 1):
        model.add(Dense(hidden_units, activation='relu'))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    
    # Выходной слой (регрессия)
    model.add(Dense(1, activation='linear'))
    
    # Компиляция модели
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_neural_network(model, X_train, y_train, X_val, y_val, epochs, batch_size, patience):
    """Обучение нейронной сети"""
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )
    
    return history

# Заголовок приложения
st.title("🏠 California House Price Prediction App")
st.markdown("Нейросетевая модель для предсказания стоимости домов в Калифорнии")

# Боковая панель для навигации
st.sidebar.title("Навигация")
page = st.sidebar.selectbox("Выберите режим", ["📊 Обучение модели", "🔮 Предсказание"])

if page == "📊 Обучение модели":
    st.header("🎓 Обучение нейросетевой модели")
    
    # Загрузка данных
    st.subheader("1. Загрузка данных")
    uploaded_file = st.file_uploader("Загрузите CSV файл с данными California Housing", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Файл успешно загружен! Размер данных: {df.shape}")
        
        # Проверка наличия необходимых столбцов
        required_columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 
                          'AveOccup', 'Latitude', 'Longitude', 'MedHouseValue']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"❌ В данных отсутствуют необходимые столбцы: {missing_columns}")
            st.stop()
        
        # Просмотр данных
        st.subheader("2. Просмотр данных")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Первые 5 строк:**")
            st.dataframe(df[required_columns].head())
        
        with col2:
            st.write("**Статистика данных:**")
            st.dataframe(df[required_columns].describe())
        
        # Информация о признаках
        st.subheader("3. Информация о признаках")
        features_info = {
            'MedInc': 'Медианный доход в блоке (в десятках тысяч USD)',
            'HouseAge': 'Медианный возраст домов в блоке (годы)',
            'AveRooms': 'Среднее количество комнат на семью',
            'AveBedrms': 'Среднее количество спален на семью',
            'Population': 'Население блока',
            'AveOccup': 'Средняя занятость домов (людей на дом)',
            'Latitude': 'Широта блока',
            'Longitude': 'Долгота блока',
            'MedHouseValue': 'Медианная стоимость домов (в сотнях тысяч USD)'
        }
        
        for feature, description in features_info.items():
            st.write(f"**{feature}**: {description}")
        
        # Проверка данных
        st.subheader("4. Проверка данных")
        if df[required_columns].isnull().any().any():
            st.warning("⚠️ В данных есть пропущенные значения. Они будут заполнены медианами.")
            df[required_columns] = df[required_columns].fillna(df[required_columns].median())
        
        # Настройка нейросети
        st.subheader("5. Настройка архитектуры нейросети")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hidden_layers = st.slider("Количество скрытых слоев:", 1, 5, 3)
            hidden_units = st.slider("Количество нейронов в слое:", 32, 256, 128)
        
        with col2:
            dropout_rate = st.slider("Dropout rate:", 0.0, 0.5, 0.2, 0.1)
            learning_rate = st.selectbox("Learning rate:", 
                                       [1e-2, 1e-3, 1e-4, 1e-5], 
                                       index=1)
        
        with col3:
            epochs = st.slider("Максимальное количество эпох:", 50, 500, 150)
            batch_size = st.selectbox("Batch size:", [16, 32, 64, 128], index=2)
            patience = st.slider("Patience для early stopping:", 5, 20, 10)
        
        # Настройка разделения данных
        st.subheader("6. Настройка обучения")
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Размер тестовой выборки (%):", 
                                10, 40, 20) / 100
        
        with col2:
            validation_size = st.slider("Размер валидационной выборки (%):", 
                                      10, 30, 15) / 100
        
        # Кнопка обучения
        if st.button("🎯 Обучить нейросеть", type="primary"):
            with st.spinner("Обучение нейросети... Это может занять несколько минут"):
                try:
                    # Подготовка данных
                    feature_columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                                     'Population', 'AveOccup', 'Latitude', 'Longitude']
                    target_column = 'MedHouseValue'
                    
                    X = df[feature_columns].values
                    y = df[target_column].values
                    
                    # Разделение на train/test
                    X_temp, X_test, y_temp, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    # Разделение на train/validation
                    val_size_adjusted = validation_size / (1 - test_size)
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_temp, y_temp, test_size=val_size_adjusted, random_state=42
                    )
                    
                    # Масштабирование данных
                    scaler_x = StandardScaler()
                    X_train_scaled = scaler_x.fit_transform(X_train)
                    X_val_scaled = scaler_x.transform(X_val)
                    X_test_scaled = scaler_x.transform(X_test)
                    
                    scaler_y = StandardScaler()
                    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
                    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
                    
                    # Создание и обучение модели
                    model = create_neural_network(
                        input_dim=len(feature_columns),
                        hidden_layers=hidden_layers,
                        hidden_units=hidden_units,
                        dropout_rate=dropout_rate,
                        learning_rate=learning_rate
                    )
                    
                    # Обучение
                    history = train_neural_network(
                        model=model,
                        X_train=X_train_scaled,
                        y_train=y_train_scaled,
                        X_val=X_val_scaled,
                        y_val=y_val_scaled,
                        epochs=epochs,
                        batch_size=batch_size,
                        patience=patience
                    )
                    
                    # Предсказания и оценка
                    y_pred_scaled = model.predict(X_test_scaled, verbose=0).flatten()
                    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                    
                    # Метрики
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    
                    metrics = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
                    
                    # Сохранение в session state
                    st.session_state.model = model
                    st.session_state.model_trained = True
                    st.session_state.scaler_x = scaler_x
                    st.session_state.scaler_y = scaler_y
                    st.session_state.feature_columns = feature_columns
                    st.session_state.history = history.history
                    st.session_state.metrics = metrics
                    st.session_state.y_test = y_test
                    st.session_state.y_pred = y_pred
                    
                    st.success("✅ Нейросеть успешно обучена!")
                    
                except Exception as e:
                    st.error(f"❌ Ошибка при обучении: {str(e)}")
            
            # Визуализация результатов
            if st.session_state.model_trained:
                st.subheader("7. Результаты обучения")
                
                # Графики обучения
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**График потерь (Loss):**")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(st.session_state.history['loss'], label='Training Loss')
                    ax.plot(st.session_state.history['val_loss'], label='Validation Loss')
                    ax.set_xlabel('Эпохи')
                    ax.set_ylabel('Loss')
                    ax.set_title('История обучения')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                with col2:
                    st.write("**График MAE:**")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(st.session_state.history['mae'], label='Training MAE')
                    ax.plot(st.session_state.history['val_mae'], label='Validation MAE')
                    ax.set_xlabel('Эпохи')
                    ax.set_ylabel('MAE')
                    ax.set_title('История MAE')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Метрики
                st.subheader("8. Метрики качества")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("MAE", f"{metrics['MAE']:.2f}")
                with col2:
                    st.metric("RMSE", f"{metrics['RMSE']:.2f}")
                with col3:
                    st.metric("MSE", f"{metrics['MSE']:.2f}")
                with col4:
                    st.metric("R² Score", f"{metrics['R2']:.4f}")
                
                # График предсказаний vs реальности
                st.subheader("9. Предсказания vs Реальность")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(y_test, y_pred, alpha=0.6)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax.set_xlabel('Реальные значения (MedHouseValue)')
                ax.set_ylabel('Предсказанные значения (MedHouseValue)')
                ax.set_title('Предсказания vs Реальность')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Распределение ошибок
                st.subheader("10. Распределение ошибок")
                errors = y_test - y_pred
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
                ax.axvline(x=0, color='red', linestyle='--', label='Zero Error')
                ax.set_xlabel('Ошибка предсказания')
                ax.set_ylabel('Частота')
                ax.set_title('Распределение ошибок предсказания')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

else:  # Страница предсказания
    st.header("🔮 Предсказание стоимости дома")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Сначала обучите модель на странице 'Обучение модели'")
        st.info("Перейдите в раздел обучения, чтобы загрузить данные и обучить нейросеть")
    else:
        st.success("✅ Нейросеть готова для предсказаний!")
        
        # Информация о модели
        st.subheader("Информация о модели")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Архитектура", "Нейросеть")
        with col2:
            st.metric("Количество признаков", len(st.session_state.feature_columns))
        with col3:
            st.metric("R² Score", f"{st.session_state.metrics['R2']:.4f}")
        
        # Форма для ввода данных
        st.subheader("Введите параметры дома:")
        
        input_data = {}
        cols = st.columns(2)
        
        # Диапазоны значений на основе реальных данных California Housing
        feature_ranges = {
            'MedInc': (0.5, 15.0, 3.0, 0.1),
            'HouseAge': (1.0, 52.0, 28.0, 1.0),
            'AveRooms': (2.0, 10.0, 5.0, 0.1),
            'AveBedrms': (0.5, 4.0, 2.0, 0.1),
            'Population': (100.0, 5000.0, 1500.0, 50.0),
            'AveOccup': (1.0, 8.0, 3.0, 0.1),
            'Latitude': (32.5, 42.0, 36.0, 0.1),
            'Longitude': (-124.5, -114.0, -119.0, 0.1)
        }
        
        for i, feature in enumerate(st.session_state.feature_columns):
            with cols[i % 2]:
                if feature in feature_ranges:
                    min_val, max_val, default_val, step = feature_ranges[feature]
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(default_val),
                        step=step,
                        help=f"Рекомендуемый диапазон: {min_val} - {max_val}"
                    )
        
        # Кнопка предсказания
        if st.button("🎯 Предсказать стоимость дома", type="primary"):
            try:
                # Подготовка входных данных
                input_df = pd.DataFrame([input_data])
                
                # Масштабирование
                input_scaled = st.session_state.scaler_x.transform(input_df)
                
                # Предсказание
                prediction_scaled = st.session_state.model.predict(input_scaled, verbose=0)
                prediction = st.session_state.scaler_y.inverse_transform(prediction_scaled)[0][0]
                
                # Отображение результата (MedHouseValue в сотнях тысяч USD)
                st.success(f"### 🏠 Предсказанная стоимость: **${prediction*100000:,.2f}**")
                st.info(f"💡 MedHouseValue: **{prediction:.2f}** (в сотнях тысяч USD)")
                
                # Дополнительная информация
                st.subheader("📊 Дополнительная информация")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Введенные параметры:**")
                    for feature, value in input_data.items():
                        st.write(f"- **{feature}**: {value}")
                
                with col2:
                    st.write("**Точность модели:**")
                    for metric, value in st.session_state.metrics.items():
                        st.write(f"- **{metric}**: {value:.4f}")
                        
            except Exception as e:
                st.error(f"❌ Ошибка при предсказании: {str(e)}")

# Футер
st.sidebar.markdown("---")
st.sidebar.info(
    "🔬 Это приложение использует нейронные сети для предсказания "
    "стоимости домов в Калифорнии на основе данных California Housing Dataset."
)