## Описание
Классификация медицинских изображений (снимков глаза) для выявления диабетической ретинопатии.

## Требования
Для запуска проекта установите зависимости, указанные в файле `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Датасет
Используемый датасет загружается с Kaggle:
- [Eye Disease Image Dataset](https://www.kaggle.com/datasets/ruhulaminsharif/eye-disease-image-dataset)

## Запуск обучения
1. Клонируйте репозиторий:
   ```bash
   git clone <repository-url>
   cd Blindness Detection
   ```
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
3. Запустите Jupyter Notebook `training.ipynb` для выполнения всех этапов проекта.

## Запуск приложения
Для запуска приложения Streamlit выполните следующую команду в терминале:

```bash
streamlit run app.py
```