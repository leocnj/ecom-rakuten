from simpletransformers.classification import ClassificationModel
import pandas as pd
from . import data
from ecom.scoring import score


def csv_to_pd(csv_file, cenc, nrow=None):
    sep = '\t' if csv_file.endswith('.tsv') else ','
    df = pd.read_csv(csv_file, sep=sep)
    df.columns = ['text', 'label']
    df.label = cenc.encode(df.label)
    print(df.head())
    return df.head(nrow) if nrow else df 


def wt_f1(real, pred):
    return score(real, pred)[0]  # only return f1


_, cenc = data.load_encoders()
df_train = csv_to_pd('data/train.csv', cenc)
df_val = csv_to_pd('data/val.csv', cenc, nrow=256)
df_eval = csv_to_pd('data/rdc-catalog-gold.tsv', cenc) 
n_label = len(cenc.itos)

args = {'fp16': False,
        'output_dir': 'albert_v2',
        'n_gpu': 8,
        'evaluate_during_training': True,
        'train_batch_size': 64,
        'eval_batch_size': 64}

# You can set class weights by using the optional weight argument
model = ClassificationModel(
    'albert', 'albert-xlarge-v2', num_labels=n_label, args=args)

# Train the model
model.train_model(df_train, eval_df=df_val)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(
    df_eval, wt_f1=wt_f1)
print(result)
