# reviews-classification

Classification of product reviews using RuBERT, XLM-RoBERTa, CatBoost, and LoRA, including data augmentation with back translation, to improve performance on minor classes.

## Общая архитектура решения

![Общая схема работы](/images/photo_2025-09-17_23-55-04.jpg)
*Рис. 1: Мыслительный роадмап решений*

## Процесс разметки данных

### Этап 0: Начальное разделение
- Изначальный train разделен на **train_df (1717 samples)** и **val_df (100 samples)**
- val_df размечен вручную для валидации

### Этап 1: Эксперименты с LLM (отвергнуто)
- **Попытка 1**: Llama2 7B - не помещается в память Google Colab ❌
- **Попытка 2**: Flan-T5-large - ответы оторваны от реальности ❌  
- **Попытка 3**: Helsinki-NLP/opus-mt-ru-en + Flan-T5-large (zero-shot) - плохая семантика ❌

### Этап 2: Первые рабочие подходы
- **Попытка 4**: Helsinki-NLP/opus-mt-ru-en → Flan-T5-base эмбеддинги → CatBoost
  - **Результат**: 70% accuracy на 50 случайных примерах
  - **Проблема**: потеря семантической информации при переводе

![Неудачные попытки](images/photo_2025-09-18_00-05-27.jpg)
*Рис. 2: Попытки 1–4 на роадмапе*

### Этап 3: Прорыв с русскими эмбеддингами
- **Попытка 5**: Эмбеддинги на DeepPavlov/rubert-base-cased ИЛИ ai-forever/sbert_large_nlu_ru → CatBoost
  - **Результат**: 70-80% accuracy на 50 случайных примерах ✅
  - **Выявленная проблема**: В valid не хватает классов "обувь", "текстиль", "товары для детей", "украшения и аксессуары", "электроника", "бытовая техника", "посуда"

### Этап 4: Активное дообучение и ансамбль
**Методология RuBERT + CatBoost с итеративной разметкой:**

1. **Первоначальное обучение** на размеченном val_df
2. **Ручная разметка** части train (десятки неразмеченных объектов)
3. **Тестирование** на train с выявлением сложных случаев
4. **Дополнительная ручная разметка** проблемных примеров
5. **Валидация с XLM-RoBERTa** (joeddav/xlm-roberta-large-xnli):
   - Сравнение расхождений между нашей моделью и zero-shot классификатором
   - Ручная коррекция спорных случаев с помощью ансамблевого алгоритма:

```python
final_labels_train = []

for _, row in train_after_hand_classification.iterrows():
    rubert = row['pseudo_label_rubert']
    xlm = row['pred_label']
    agree = row['agree?']

    if rubert == xlm:
        final_labels_train.append(rubert)
    else:
        if 'текстиль' in [rubert, xlm]:
            if agree == False:
                final_labels_train.append(xlm)
            else:
                final_labels_train.append(rubert)
        elif 'товары для детей' in [rubert, xlm]:
            # Всегда берём RuBERT+CB
            final_labels_train.append(rubert)
        else:
            # Остальные классы — берём RuBERT+CB
            final_labels_train.append(rubert)

train_after_hand_classification['pseudo_label_ensemble_final'] = final_labels_train
