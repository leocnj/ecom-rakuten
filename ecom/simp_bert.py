from simpletransformers.classification import ClassificationModel
import pandas as pd
from . import data

def csv_to_pd(csv_file, cenc):
    df = pd.read_csv(csv_file)
    df.columns = ['text', 'label']
    df.label = cenc.encode(df.label)
    print(df.head())
    return df

_, cenc = data.load_encoders()
df_train = csv_to_pd('data/train.csv', cenc)
df_val = csv_to_pd('data/val.csv', cenc)
n_label = len(cenc.itos)

model = ClassificationModel('roberta', 'roberta-base', num_labels=n_label, args={'reprocess_input_data': True, 'fp16': False, 'n_gpu': 4}) # You can set class weights by using the optional weight argument

# Train the model
model.train_model(df_train)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(df_val)
