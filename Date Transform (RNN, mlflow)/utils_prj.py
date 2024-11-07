import random

import numpy as np
import tensorflow as tf
from babel.dates import format_date
from faker import Faker
from keras.layers import (
    LSTM,
    Activation,
    Bidirectional,
    Concatenate,
    Dense,
    Dot,
    Input,
    RepeatVector,
    TextVectorization,
)
from keras.models import Model
from keras.utils import to_categorical
from tqdm import tqdm


def generate_date():
    """
    _summary_

    Returns
    -------
    _type_
        _description_
    """
    # Define format of the data we would like to generate
    FORMATS = ['short', 'medium', 'long'] \
            + ['full']*10 \
            + ['d MMM YYY', 'd MMMM YYY', 'dd MMM YYY', 'd MMM, YYY', 'd MMMM, YYY', 'dd, MMM YYY', 'd MM YY', 'd MMMM YYY', 'MMMM d YYY', 'MMMM d, YYY', 'dd.MM.YY']

    faker = Faker()
    dt = faker.date_object()

    try:
        human_readable = format_date(dt, format=random.choice(FORMATS),  locale='en_US') # locale=random.choice(LOCALES))
        human_readable = human_readable.lower()
        human_readable = human_readable.replace(',','')
        machine_readable = dt.isoformat()
    except AttributeError:
        return None, None, None

    return human_readable, machine_readable, dt


def load_dataset(m):
    """
    _summary_

    Parameters
    ----------
    m : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    dataset = []
    for i in tqdm(range(m)):
        h, m, _ = generate_date()
        if h is not None:
            dataset.append((h, m))

    # human = dict(zip(sorted(human_vocab) + ['<unk>', '<pad>'], 
    #                  list(range(len(human_vocab) + 2))))
    # inv_machine = dict(enumerate(sorted(machine_vocab)))
    # machine = {v:k for k,v in inv_machine.items()}

    return dataset


def process_data(dataset, Tx, Ty, verbose=False):
    """
    Transform source dataset from date string representation to one-hot encoded.

    Parameters
    ----------
    dataset : list
        An array of tuples with human date and corresponding machine date
    """

    X, y = zip(*dataset)

    X = np.array(X)
    y = np.array(y)
    print(X.shape)

    tokenizer_X = TextVectorization(split='character', standardize='lower')
    tokenizer_y = TextVectorization(split='character', standardize='lower')

    tokenizer_X.adapt(X)
    tokenizer_y.adapt(y)

    # Get the vocabulary (list of tokens)
    vocab_X = tokenizer_X.get_vocabulary()
    vocab_y = tokenizer_y.get_vocabulary()
    # Create the mapping from character to index
    char_to_index_X = {char: index for index, char in enumerate(vocab_X)}
    char_to_index_y = {char: index for index, char in enumerate(vocab_y)}
    # Create the mapping from index to character
    index_to_char_X = {index: char for index, char in enumerate(vocab_X)}
    index_to_char_y = {index: char for index, char in enumerate(vocab_y)}

    _, _, X_tokenized_pad_oh = prepare_X(X=X,X_vis='X', tokenizer=tokenizer_X, max_len=Tx, vocab=char_to_index_X, idx_vis=5, verbose=verbose)
    _, _, y_tokenized_pad_oh = prepare_X(X=y, X_vis='y', tokenizer=tokenizer_y, max_len=Ty, vocab=char_to_index_y, idx_vis=5, verbose=verbose)

    if verbose:
        print('X.shape: ', X.shape)
        print('y.shape: ', y.shape)
        print(f'char_to_index_X ({len(char_to_index_X)}): ', char_to_index_X)
        print(f'char_to_index_y ({len(char_to_index_y)}): ', char_to_index_y)
        print(f'index_to_char_X ({len(index_to_char_X)}): ', index_to_char_X)
        print(f'index_to_char_y ({len(index_to_char_y)}): ', index_to_char_y)   

    return X_tokenized_pad_oh, y_tokenized_pad_oh, char_to_index_X, char_to_index_y, index_to_char_X, index_to_char_y, tokenizer_X, tokenizer_y


def prepare_X(X, X_vis, tokenizer, max_len, vocab, idx_vis, verbose=False):
    """
    Convert the input array of strings X into one-hot encoded chars of these strings

    Parameters
    ----------
    X : _type_
        _description_
    X_name : string
        Name of the variable X. Use only for verbose output visualization.
    tokenizer : TextVectorization
        Tokenizer used for tokenizing training data.
    Tx : max length of strings
        _description_
    vocab : _type_
        _description_
    verbose : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    # tokenize X (convert chars to numbers)
    X_tokenized = np.array(tokenizer(X))

    # pad sequences till Tx if length of string is less than Tx, otherwise - cut it till Tx
    X_tokenized_pad = np.array([np.pad(s, pad_width=(0, max_len-len(s))) if len(s) < max_len else s[:len(s)] for s in X_tokenized])

    # one-hot encode numbers
    X_tokenized_pad_oh = np.array([to_categorical(s, num_classes=len(vocab)) for s in X_tokenized_pad])

    if verbose:
        print(f'For variable {X_vis}')
        print(f'{X_vis}_tokenized.shape: ', X_tokenized.shape)
        print(f'{X_vis}_tokenized_pad.shape: ', X_tokenized_pad.shape)
        print(f'{X_vis}_tokenized_pad_oh.shape: ', X_tokenized_pad_oh.shape)
        print(f'{X_vis}_vocab: ', vocab)
        print(f'for idx = {idx_vis}')
        print(f'{X_vis}[{idx_vis}]: ', X[idx_vis])
        print(f'{X_vis}_tokenized[{idx_vis}]: ', X_tokenized[idx_vis])
        print(f'{X_vis}_tokenized_pad[{idx_vis}]: ', X_tokenized_pad[idx_vis])
        print(f'{X_vis}_tokenized_pad_oh[{idx_vis}]: ', X_tokenized_pad_oh[idx_vis])
        print()

    return X_tokenized, X_tokenized_pad, X_tokenized_pad_oh


def softmax_custom(x, axis=1):
    """Softmax activation function.

    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied (default is -1).

    # Returns
        Tensor, output of softmax transformation.

    # Raises
        ValueError: In case `dim(x) == 1`.
    """

    ndim = x.shape[1]
    if ndim == 2:
        return tf.nn.softmax(x)
    elif ndim > 2:
        e = tf.exp(x - tf.reduce_max(x, axis=axis, keepdims=True))  # Use TensorFlow's exp
        s = tf.reduce_sum(e, axis=axis, keepdims=True)  # Use TensorFlow's sum

        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')


def build_model_rnn(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """

    def one_step_attention(a, s_prev):
        """
        Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
        "alphas" and the hidden states "a" of the Bi-LSTM.
        
        Arguments:
        a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
        s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
        
        Returns:
        context -- context vector, input of the next (post-attention) LSTM cell
        """
        
        # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
        s_prev = repeator(s_prev)
        # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
        # For grading purposes, please list 'a' first and 's_prev' second, in this order.
        concat = concatenator([a, s_prev])
        # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
        e = densor1(concat)
        # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
        energies = densor2(e)
        # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
        alphas = activator(energies)
        # Use dotor together with "alphas" and "a", in this order, to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
        context = dotor([alphas, a])

        return context, alphas, energies, e

    # Defined shared layers as global variables
    repeator = RepeatVector(Tx)
    concatenator = Concatenate(axis=-1)
    densor1 = Dense(10, activation = "tanh")
    densor2 = Dense(1, activation = "tanh")
    activator = Activation(softmax_custom, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
    dotor = Dot(axes = 1)
    # Please note, this is the post attention LSTM cell.  
    post_activation_LSTM_cell = LSTM(n_s, return_state = True) # Please do not modify this global variable.
    output_layer = Dense(len(char_to_index_y), activation=softmax_custom)


    # Define the inputs of your model with a shape (Tx,)
    # Define s0 (initial hidden state) and c0 (initial cell state)
    # for the decoder LSTM with shape (n_s,)
    X = Input(shape=(Tx, len(char_to_index_X)))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0

    # Initialize empty list of outputs
    outputs = []

    # Step 1: Define your pre-attention Bi-LSTM. (≈ 1 line)
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)

    # Step 2: Iterate for Ty steps
    attention_weights = []
    energies_weights = []
    e_weights = []
    for t in range(Ty):
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
        context, alphas, energies, e = one_step_attention(a, s)

        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])

        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        out = output_layer(s)

        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)
        attention_weights.append(alphas)
        energies_weights.append(energies)
        e_weights.append(e)

    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model = Model(inputs=[X, s0, c0],outputs=outputs)

    attention_model = Model(inputs=[X, s0, c0], outputs=attention_weights)
    attention_inspector = Model(inputs=[X, s0, c0], outputs=[attention_weights, energies_weights, e_weights])

    return model, attention_model, attention_inspector


def predict_model(model, date, n_s, index_to_char_y, tokenizer_X, tokenizer_y, char_to_index_X, Tx, verbose=False):
    """
    Return translation of the human written date to unified machine date format.

    Parameters
    ----------
    date : str or list or np.ndarray
        A date string or an array of date strings

    Returns
    -------
    A date string or an array of date strings in unified machine format.
    """
    print('*** predict_model *** ')
    
    assert isinstance(date, np.ndarray) or isinstance(date, list) or isinstance(date, str), 'date MUST be a string of date or an array of date strings.'
    
    if isinstance(date, str):
        print(type(date))
        date=np.array([date])
        print(type(date))
        print(date)
        print(len(date))
        
    s0 = np.zeros((len(date),n_s))
    c0 = np.zeros((len(date),n_s))
    # s0 = np.zeros(1,n_s)
    # c0 = np.zeros(1,n_s)

    # prepare test examples
    _,_, X_tokenized_pad_oh_test = prepare_X(X=date, X_vis='X', tokenizer=tokenizer_X, max_len=Tx, vocab=char_to_index_X, idx_vis=0, verbose=verbose)

    #print('X_tokenized_pad_oh_test.shape', X_tokenized_pad_oh_test.shape)
    #print('s0, c0: ',  s0, c0)

    # predict
    pred_softmax = model.predict([X_tokenized_pad_oh_test, s0, c0], verbose=verbose)
    pred_num = np.argmax(pred_softmax, axis=-1)
    pred_char = [[index_to_char_y[n] for n in e] for e in pred_num]
    pred_joined = [''.join(e) for e in np.array(pred_char).T]

    result_df = pd.DataFrame({'test_examples': date, 'pred': pred_joined})

    return result_df

def clean_history(model_history):

    history_df = pd.DataFrame(model_history.history)

    history_df.columns = history_df.columns.str.replace(r'^dense_[0-9]{1,3}', 'train_last_layer', regex=True)
    history_df.columns = history_df.columns.str.replace(r'^val_dense_[0-9]{1,3}', 'val_last_layer', regex=True)
    history_df.columns = history_df.columns.str.replace('train_last_layer_accuracy', 'train_accuracy', regex=False)
    history_df.columns = history_df.columns.str.replace('val_last_layer_accuracy', 'val_accuracy', regex=False)
    history_df.rename({'loss': 'train_loss'}, axis=1, inplace=True)


    #history_df.columns = history_df.columns.str.replace(r'^val_dense_[0-9]{1,3}_accuracy$', 'val_accuracy', regex=True)
    # history_df.columns = history_df.columns.str.replace(r'^dense_[0-9]{1,3}_accuracy$', 'train_accuracy', regex=True)
    # #print('2', history_df.columns)
    # history_df.columns = history_df.columns.str.replace(r'^dense_[0-9]{1,3}_loss$', 'train_loss', regex=True)
    # history_df.columns = [re.sub(r'^val_dense_[0-9]{1,3}', 'val', col) for col in history_df.columns]
    # history_df.columns = [re.sub(r'dense_[0-9]{1,3}_', 'train_', col) for col in history_df.columns]
    # #print('3', history_df.columns)

    return history_df