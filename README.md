# Customer Churn Prediction — Bank

Кратко:
Построен end-to-end ML-пайплайн для предсказания оттока клиентов (таргет 'Exited') с использованием Python и CatBoost.
Данные: Customer-Churn-Records.csv

Что сделано:
- Предобработка: очистка, кодирование категорий, заполнение пропусков.
- Модель: CatBoostClassifier (iterations=200).
- Оценка: классификационный отчёт, ROC AUC, confusion matrix.
- Экспорт: прогнозы и вероятности для Power BI (data/churn_predictions_export.csv).
- Сохранены модель и графики (data/model_catboost.joblib, data/figures/*.png).

Как запустить:
1. Поместите `Customer-Churn-Records.csv` рядом с `main.py`.
2. Установите зависимости:
   `pip install pandas numpy scikit-learn catboost matplotlib seaborn joblib`
3. Запустите:
   `python main.py`

Автор: Tezekbay Almara
