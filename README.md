# 📝 Reviews Classification  

**Задача**: классификация товарных отзывов по категориям с использованием  
**RuBERT, XLM-RoBERTa, CatBoost и LoRA**.  

Для улучшения качества на редких классах применялась:  
- **аугментация через back-translation**,  
- **итеративная ручная разметка порядка десятка примеров**.  

---

## 📌 Архитектура решения  

![Общая схема работы](/images/photo_2025-09-17_23-55-04.jpg)  
*Рис. 1: Мыслительный роадмап решений*  

---

## 🏷️ Процесс разметки данных  

### 🔹 Этап 0: Начальное разделение
- **train_df** → 1717 samples  
- **val_df** → 100 samples (ручная разметка)  

---

### 🔹 Этап 1: Эксперименты с LLM (❌ неудачные)
1. Llama2 7B (zero-shot) — не помещается в Colab  
2. Flan-T5-large (zero-shot) — "галлюцинации"  
3. Helsinki-NLP/opus-mt-ru-en + Flan-T5-large — слабая семантика  

![Неудачные попытки](images/photo_2025-09-18_00-05-27.jpg)  
*Рис. 2: Попытки 1–3 на роадмапе*  

---

### 🔹 Этап 2: Первые рабочие подходы  
- **Helsinki-NLP/opus-mt-ru-en + Flan-T5-base** → эмбеддинги → CatBoost  
- Accuracy ≈ **70%** на 50 случайных примерах  
- ⚠️ Потеря семантики при переводе  

---

### 🔹 Этап 3: Прорыв с русскими эмбеддингами ✅  
- **DeepPavlov/rubert-base-cased**  
- **ai-forever/sbert_large_nlu_ru** → CatBoost  
- Accuracy ≈ **70–80%**  
- Проблема: дисбаланс классов → решено через `cleaning_datasets.ipynb`  

---

### 🔹 Этап 4: Активное дообучение и ансамбль  

**Методология RuBERT + CatBoost**:  
1. Первичное обучение на `valid`  
2. Ручная разметка части `train`  
3. Тестирование и поиск сложных кейсов  
4. Дополнительная ручная разметка  
5. **Ансамбль RuBERT + XLM-RoBERTa (zero-shot)** → согласование меток  

Пример логики ансамблирования:  

```python
if rubert == xlm:
    final = rubert
elif "текстиль" in [rubert, xlm]:
    final = xlm if not agree else rubert
elif "товары для детей" in [rubert, xlm]:
    final = rubert
else:
    final = rubert
```
В итоге получаем размеченные `train`, `valid`, `test`. Время разметки `test` — 13:02 (13 минут, 2 секунды)

## 🔄 Аугментация редких классов (Back-Translation)

Используется back-translation через случайный язык из `intermediate_languages`.  
Полученные тексты добавляются в `train` для увеличения выборки редких классов.

```python
import random
import time
from deep_translator import GoogleTranslator

def augment_text(text, variations=3):
    """
    Функция аугментации текста через back-translation.
    """
    augmented_texts = []
    intermediate_languages = ['en', 'de', 'fr', 'es', 'it']

    for i in range(variations):
        try:
            # Случайный выбор промежуточного языка
            intermediate_lang = random.choice(intermediate_languages)

            # Перевод текста на промежуточный язык
            translated_intermediate = GoogleTranslator(
                source='ru',
                target=intermediate_lang
            ).translate(text)

            time.sleep(0.1)

            # Обратный перевод на русский
            back_translated = GoogleTranslator(
                source=intermediate_lang,
                target='ru'
            ).translate(translated_intermediate)

            augmented_texts.append(back_translated)
            time.sleep(0.2)

        except Exception as e:
            print(f"Ошибка при аугментации: {e}")
            augmented_texts.append(text)

    return augmented_texts

train_df_augmented = pd.concat([train, augmented_df], ignore_index=True)
train_df_augmented.to_csv('train_split_augmented.csv', index=False)

```
Полученный новый train_df_augmented был разбит на `train_df` и `valid_df` со стратификацией по классам, чтобы сохранить масштаб

## 🤖 Обучение модели с LoRA на аугментированных данных

Здесь мы используем **LoRA (Low-Rank Adaptation)** для дообучения RuBERT на аугментированных данных.

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch
import pickle
from sklearn.metrics import f1_score
import numpy as np

# Загружаем аугментированные train и validation данные
dataset = load_dataset('csv', data_files={
    'train': '/content/train_df_aug.csv', 
    'validation': '/content/valid_df_aug.csv'
})

tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

# Кодирование меток
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(dataset['train']['pseudo_label_ensemble_final'])
valid_labels = label_encoder.transform(dataset['validation']['pseudo_label_ensemble_final'])

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

dataset['train'] = dataset['train'].add_column('label', train_labels)
dataset['validation'] = dataset['validation'].add_column('label', valid_labels)

# Токенизация текста
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['text', 'pseudo_label_ensemble_final'])
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')

# Вычисление весов классов для балансировки
classes = np.array(train_labels)
class_weights = compute_class_weight('balanced', classes=np.unique(classes), y=classes)
class_weights = torch.tensor(class_weights, dtype=torch.float).to('cuda')

# Инициализация модели и LoRA
model = AutoModelForSequenceClassification.from_pretrained(
    'DeepPavlov/rubert-base-cased',
    num_labels=6,
    device_map='cuda'
)

lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=['query', 'value'],
    lora_dropout=0.4,
    bias='none',
    task_type='SEQ_CLS'
)
model = get_peft_model(model, lora_config)

# Кастомный Trainer с учетом весов классов
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Функция метрик (F1 по классам и взвешенный)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    weighted_f1 = f1_score(labels, predictions, average='weighted')
    per_class_f1 = f1_score(labels, predictions, average=None)
    class_names = label_encoder.classes_
    for cls, f1 in zip(class_names, per_class_f1):
        print(f"Class {cls}: F1 = {f1}")
        with open('metrics.txt', 'a') as f:
            f.write(f"Epoch {trainer.state.epoch}: Class {cls}: F1 = {f1}\n")
    with open('metrics.txt', 'a') as f:
        f.write(f"Epoch {trainer.state.epoch}: Weighted F1 = {weighted_f1}\n")
    print(f"Epoch {trainer.state.epoch}: Weighted F1 = {weighted_f1}")
    return {'weighted_f1': weighted_f1}

# Настройки обучения
training_args = TrainingArguments(
    output_dir='./results_lora_1',
    num_train_epochs=20,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    fp16=True,
    seed=42
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics
)

# Отключение WANDB и запуск обучения
import os
os.environ["WANDB_DISABLED"] = "true"
import wandb
wandb.init(mode="disabled")

trainer.train()
```

## 🔹 Получение лучшей модели и предсказание теста

```python
from peft import PeftModel
import torch
import pickle
import time
from tqdm import tqdm
from sklearn.metrics import f1_score
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Загружаем токенизатор и модель
tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
checkpoint_path = '/content/results_lora_1/checkpoint-1632'
model = AutoModelForSequenceClassification.from_pretrained(
    'DeepPavlov/rubert-base-cased',
    num_labels=6
)
model = PeftModel.from_pretrained(model, checkpoint_path)
model = model.to('cuda')

# Загружаем label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Загружаем тестовый датасет
test_df = pd.read_csv('/content/drive/MyDrive/NLP_CASE_FINAL/FINAL_TEST.csv')
test_dataset = load_dataset('csv', data_files={'test': '/content/drive/MyDrive/NLP_CASE_FINAL/FINAL_TEST.csv'})

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

tokenized_test = test_dataset['test'].map(tokenize_function, batched=True)
tokenized_test = tokenized_test.remove_columns(['text'])

# Предсказания
model.eval()
predictions = []
start_time = time.time()

for i in tqdm(range(0, len(tokenized_test), 16)):
    batch = tokenized_test[i:i+16]
    inputs = {
        'input_ids': torch.tensor(batch['input_ids']).to('cuda'),
        'attention_mask': torch.tensor(batch['attention_mask']).to('cuda')
    }
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
    predictions.extend(batch_preds)

# Среднее время предсказания
elapsed_time = time.time() - start_time
avg_time_per_example = elapsed_time / len(test_df)
print(f"Average inference time per example: {avg_time_per_example:.3f} seconds")

# Обратное преобразование меток
predictions = label_encoder.inverse_transform(predictions)
predictions = ['текстиль' if pred == 'обувь' else pred for pred in predictions]

# Метрики
ground_truth = test_df['pseudo_label_ensemble_final'].values
weighted_f1 = f1_score(ground_truth, predictions, average='weighted')
per_class_f1 = f1_score(ground_truth, predictions, average=None)
class_names = sorted(test_df['pseudo_label_ensemble_final'].unique())
for cls, f1 in zip(class_names, per_class_f1):
    print(f"Class {cls}: F1 = {f1}")
print(f"Weighted F1-score on test set: {weighted_f1}")

# Сохранение результатов
test_df['predicted_label'] = predictions
test_df[['Unnamed: 0', 'text', 'pseudo_label_ensemble_final', 'predicted_label']].to_csv('test_predictions.csv', index=False)
with open('test_metrics.txt', 'w') as f:
    for cls, f1 in zip(class_names, per_class_f1):
        f.write(f"Class {cls}: F1 = {f1}\n")
    f.write(f"Weighted F1-score on test set: {weighted_f1}\n")
print("Predictions saved to test_predictions.csv")
print("Metrics saved to test_metrics.txt")
```

## 📊 Результаты

- Среднее время предсказания класса объекта в `test`: 0.031 seconds
- Weighted F1-score на `test`: 0.6983976359503813
- Все метрики сохранены в `test_metrics.txt` для `test` и в `metrics.txt` для `LoRA`
