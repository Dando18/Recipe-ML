'''
Daniel Nichols
'''

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import ast
import itertools
import os
from argparse import ArgumentParser


def load_data(args):
    print('=== Loading Data ===')
    data = pd.read_csv('dataset/RAW_recipes.csv', usecols=['n_steps', 'steps', 'ingredients', 'n_ingredients'], encoding='utf-8')
    print('=== Done Loading Data ===')

    print('=== Preprocessing Data ===')
    # add stop column
    data['END_INGR'] = 'END_INGR'
    data['END_STEP'] = 'END_STEP'

    # convert string columns to lists
    data['ingredients'] = data['ingredients'].apply(ast.literal_eval)
    data['steps'] = data['steps'].apply(ast.literal_eval)

    # join ingredients, END_INGR, and steps
    recipes = data['ingredients'] + data['END_INGR'].apply(lambda x: [x]) + data['steps'] + data['END_STEP'].apply(lambda x: [x])
    max_vocab_size = recipes.explode().nunique()
    vocab_size = args.vocab_size if args.vocab_size else max_vocab_size
    print('=== Done Preprocessing Data ===')

    print('=== Tokenizing Data ===')
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(recipes)
    train_sequences = tokenizer.texts_to_sequences(recipes)
    max_len = max([len(x) for x in train_sequences])
    #train_padded = pad_sequences(train_sequences, padding='post', truncating='post', maxlen=max_len)
    train_joined = list(itertools.chain.from_iterable(train_sequences))
    print('=== Done Tokenizing Data ===')

    return recipes, train_joined, tokenizer, vocab_size, max_len


def get_model(args, sequence_length, vocab_size, batch_size, use_embedding=True):
    model = Sequential()

    if use_embedding:
        model.add(Embedding(vocab_size, args.units, batch_input_shape=[batch_size, None]))
        model.add(LSTM(args.units, return_sequences=True))
    else:
        model.add(LSTM(args.units, input_shape=(sequence_length, vocab_size)))

    model.add(Dropout(0.2))
    model.add(Dense(vocab_size))
    return model


def generate(model, start, tokenizer):
    temp = 1.0
    ngen = 100

    end_steps_id = tokenizer.texts_to_sequences([['end_step']])[0][0]

    if start is None:
        input_eval = [end_steps_id]
        input_eval = tf.expand_dims(input_eval, 0)
    else:
        input_eval = tokenizer.texts_to_sequences([start])

    out = []
    last = 0

    model.reset_states()
    while last != end_steps_id:
        pred = model(input_eval)
        pred = tf.squeeze(pred, 0)

        pred = pred / temp
        predicted_id = tf.random.categorical(pred, num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        last = predicted_id
        out.append(tokenizer.sequences_to_texts([[predicted_id]]))
    
    return out


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def main():
    parser = ArgumentParser()
    parser.add_argument('-u', '--units', type=int, default=512)
    parser.add_argument('--vocab-size', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--generate', type=int)
    args = parser.parse_args()

    recipes, training_data, tokenizer, vocab_size, sequence_length = load_data(args)
    seq_length = 20

    data_set = tf.data.Dataset.from_tensor_slices(training_data)
    data_set = data_set.batch(seq_length+1, drop_remainder=True)
    data_set = data_set.map(lambda x: (x[:-1], x[1:]))

    data_set = data_set.shuffle(10000).batch(args.batch_size, drop_remainder=True)

    model = get_model(args, seq_length, vocab_size, args.batch_size, use_embedding=True)
    model.compile(loss=loss, optimizer='adam')
    model.summary()

    checkpoint_dir = './main_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

    if args.train:
        history = model.fit(data_set, epochs=args.epochs, callbacks=[checkpoint_callback])
    
    if args.generate:
        model = get_model(args, seq_length, vocab_size, 1, use_embedding=True)
        model.compile(loss=loss, optimizer='adam')
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        model.build(tf.TensorShape([1, None]))

        for _ in range(args.generate):
            out = generate(model, None, tokenizer)
            out = [x[0] for x in out if x[0] != '<OOV>']
            print("========\n")
            print('\n'.join(out))
            print('========\n\n')





if __name__ == '__main__':
    main()