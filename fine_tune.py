import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
from datasets import load_dataset, Features, Value, ClassLabel
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

LEARNING_RATE = 1e-5
BATCH_SIZE = 8

class Dataset:
    def __init__(self, train_file, test_file, batch_size):
        self.data_files = {
            'train': train_file,
            'test': test_file
        }
        self.batch_size = batch_size
        
        self.is_loaded = False
        self.dataset = None
        
        self.has_train = False
        self.train_ds = None
        self.has_test = False
        self.test_ds = None
        
        class_names = ['no spike', 'volatility spike']
        self.features = Features({'text': Value('string'), 'label': ClassLabel(names=class_names)})
    
    def load(self):
        self.is_loaded = True
        self.dataset = load_dataset('csv', data_files=self.data_files, features=self.features)
        
    def map(self, fn):
        self.dataset = self.dataset.map(fn, batched=True)
        
    def train(self, tokenizer):
        if self.has_train:
            return self.train_ds
        else:
            self.train_ds = self._format_dataset(self.dataset['train'], tokenizer)
            self.has_train = True
            return self.train_ds
    
    def test(self, tokenizer):
        if self.has_test:
            return self.test_ds
        else:
            self.test_ds = self._format_dataset(self.dataset['test'], tokenizer)
            self.has_test = True
            return self.test_ds
    
    def _format_dataset(self, ds, tokenizer):  
        ds = ds.remove_columns(['text']).with_format('tensorflow')
        features = {x: ds[x].to_tensor() for x in tokenizer.tokenizer.model_input_names}
        ds = tf.data.Dataset.from_tensor_slices((features, ds['label']))
        ds = ds.shuffle(len(ds)).batch(self.batch_size)
        return ds

class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)
    
def get_args():
    parser = ArgumentParser()
    
    parser.add_argument('--model', metavar='model', type=str, help='The HuggingFace model to fine-tune', required=True)
    parser.add_argument('--train_file', metavar='train_file', type=str, help='The CSV file containing training data', required=True)
    parser.add_argument('--test_file', metavar='test_file', type=str, help='The CSV file containing testing data', required=True)
    parser.add_argument('--epochs', metavar='epochs', type=int, help='The number of epochs to train on', required=True)
    parser.add_argument('--learning_rate', metavar='learning_rate', type=float, help='The learning rate for fine-tuning', default=LEARNING_RATE, required=False)
    parser.add_argument('--batch_size', metavar='batch_size', type=int, help='The batch size for fine-tuning', default=BATCH_SIZE, required=False)
    
    return parser.parse_args()

def main():
    args = get_args()
   
    tokenizer = Tokenizer(AutoTokenizer.from_pretrained(args.model))

    dataset = Dataset(args.train_file, args.test_file, args.batch_size)
    dataset.load()
    dataset.map(tokenizer)

    train_tf_dataset = dataset.train(tokenizer)
    test_tf_dataset = dataset.test(tokenizer)
    
    model = TFAutoModelForSequenceClassification.from_pretrained(args.model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.metrics.SparseCategoricalAccuracy(),
    )

    model.fit(train_tf_dataset, validation_data=test_tf_dataset, epochs=args.epochs)

    preds = model.predict(test_tf_dataset, batch_size=args.batch_size)
    labels = np.argmax(preds.logits, axis=-1)
    print(labels)    

if __name__=='__main__':
    main()