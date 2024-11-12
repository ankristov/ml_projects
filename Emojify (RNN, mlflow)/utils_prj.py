
import random

import emoji
import numpy as np
import pandas as pd
from keras import regularizers
from keras.layers import LSTM, Activation, Dense, Dropout, Embedding, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from nltk import download
from nltk.corpus import wordnet

# Download required resources
download('wordnet')
download('omw-1.4')  # To ensure broad coverage of synonyms


EMOJI_DICTIONARY = {"0": ":red_heart:",
                    "1": ":baseball:",
                    "2": ":grinning_face:",
                    "3": ":disappointed_face:",
                    "4": ":fork_and_knife:"}


def label_to_emoji(label):
    """
    Converts a label (int or string) into the corresponding emoji code (string) ready to be printed

    Parameters
    ----------
    label : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return emoji.emojize(EMOJI_DICTIONARY[str(label)], variant='text_type')


def load_glove_embeddings(file_path) -> tuple[dict, dict, dict]:
    """
    Load GloVe word embeddings.

    Parameters
    ----------
    file_path : str
        File path to the .txt file with embeddings.

    Returns
    -------
    _type_
        _description_
    """
    with open(file_path, 'r') as f:
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float32)

        words_to_index = {}
        index_to_words = {}
        for i, (k, v) in enumerate(word_to_vec_map.items()):
            words_to_index[k] = i
            index_to_words[i] = k

    return words_to_index, index_to_words, word_to_vec_map


def sentence_to_avg(sentence, word_to_vec_map):
    """
    _Calculate average of word embeddings for the sentence

    Parameters
    ----------
    sentence : _type_
        _description_
    word_to_vec_map : _type_
        _description_
    """

    words = sentence.lower().split()

    avg = np.zeros_like(list(word_to_vec_map.values())[0])

    for word in words:
        avg += word_to_vec_map[word]

    if len(words) > 0:
        avg = avg / len(words)
    else:
        avg = None

    return avg


def sentences_to_indices(X, word_to_index, max_len):
    
    m = X.shape[0]

    X_indices = np.zeros((m, max_len))

    for i in range(m):
        sentence_words = X[i].lower().split()
        for j, w in enumerate(sentence_words):
            if w in word_to_index:
                X_indices[i,j] = word_to_index[w]

    return X_indices


def build_model_basic(input_shape, output_shape):
    """
    Base model takes as it's input - an vector of averaged values per words embeddings in a sentence,
    push them to one dense layer, and make prediction of the smile with softmax activation.
    """

    input_avg_embeddings = Input(shape=input_shape, dtype='float32')
    X = Dense(units=output_shape, activation='softmax', kernel_regularizer = regularizers.l2(0.01))(input_avg_embeddings)

    model = Model(inputs=input_avg_embeddings, outputs=X)

    return model


# Test function
def test_model_basic(target):
    word_to_vec_map = {'a': [3, 3], 'synonym_of_a': [3, 3], 'a_nw': [2, 4], 'a_s': [3, 2], 'a_n': [3, 4], 
                       'c': [-2, 1], 'c_n': [-2, 2],'c_ne': [-1, 2], 'c_e': [-1, 1], 'c_se': [-1, 0], 
                       'c_s': [-2, 0], 'c_sw': [-3, 0], 'c_w': [-3, 1], 'c_nw': [-3, 2]}
    for key in word_to_vec_map.keys():
        word_to_vec_map[key] = np.array(word_to_vec_map[key])

    # Sample sentences and labels
    X = np.asarray(['a a_s synonym_of_a a_n c_sw', 'a a_s a_n c_sw', 'a_s  a a_n', 'synonym_of_a a a_s a_n c_sw', 
                    " a_s a_n", " a a_s a_n c ", " a_n  a c c c_e", 'c c_nw c_n c c_ne', 
                    'c_e c c_se c_s', 'c_nw c a_s c_e c_e', 'c_e a_nw c_sw', 'c_sw c c_ne c_ne'])
    Y = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    # Convert sentences to average word embeddings
    X_avg = np.array([sentence_to_avg(tweet, word_to_vec_map) for tweet in X])

    # One-hot encoding for labels
    Y_oh = to_categorical(Y, num_classes=2)

    # Get input shape and output shape
    any_word = list(word_to_vec_map.keys())[0]
    nx = word_to_vec_map[any_word].shape[0]  # Embedding size
    ny = Y_oh.shape[1]  # Number of classes

    # Build model
    np.random.seed(1)
    model = target(input_shape=(nx,), output_shape=ny)

    # Compile model with Adam optimizer
    optimizer_adam = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_adam, metrics=['accuracy'])

    # Fit model
    model.fit(X_avg, Y_oh, epochs=100, batch_size=32, shuffle=True)

    # Make predictions
    pred = model.predict(X_avg)
    pred = np.argmax(pred, axis=1)

    # Calculate accuracy
    accuracy = (pred == Y).mean()

    print("Predictions:", pred)
    print("Accuracy:", accuracy)


def prepare_XY_model_basic(X_train, Y_train, X_test, Y_test, word_to_vec_map=None, **kwargs):
    """
    _summary_

    Parameters
    ----------
    X_train : _type_
        _description_
    Y_train : _type_
        _description_
    X_test : _type_
        _description_
    Y_test : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    X_train_avg = np.array([sentence_to_avg(sentence, word_to_vec_map) for sentence in X_train])
    X_test_avg = np.array([sentence_to_avg(tweet, word_to_vec_map) for tweet in X_test])
    # X_train_avg = np.array([sentence_to_avg(sentence) for sentence in X_train])
    # X_test_avg = np.array([sentence_to_avg(tweet) for tweet in X_test])
    
    # one-hot encode Y
    num_classes = len(np.unique(Y_train))
    Y_train_oh = to_categorical(Y_train, num_classes=num_classes)
    Y_test_oh = to_categorical(Y_test, num_classes=num_classes)

    return X_train_avg, Y_train_oh, X_test_avg, Y_test_oh


def prepare_XY_model_rnn(X_train, Y_train, X_test, Y_test, word_to_index=None, max_len=None, **kwargs):
    # prepare X_train: convert words to indexes
    X_train_indices = sentences_to_indices(X_train, word_to_index, max_len)
    X_test_indices = sentences_to_indices(X_test, word_to_index, max_len)
    # X_train_indices = sentences_to_indices(X_train)
    # X_test_indices = sentences_to_indices(X_test)
    # one-hot encode Y
    num_classes = len(np.unique(Y_train))
    Y_train_oh = to_categorical(Y_train, num_classes=num_classes)
    Y_test_oh = to_categorical(Y_test, num_classes=num_classes)

    return X_train_indices, Y_train_oh, X_test_indices, Y_test_oh


def pretrained_embedding_layer(word_to_vec_map):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors from GloVe

    Parameters
    ----------
    word_to_vec_map : dict
        dictionary mapping words to their GloVe vector representation.
    """

    vocab_size = len(word_to_vec_map) + 1
    any_word = list(word_to_vec_map.keys())[0]
    emb_dim = word_to_vec_map[any_word].shape[0]

    # create embedding matrix
    emb_matrix = np.zeros([vocab_size, emb_dim])
    for i, (word, emb) in enumerate(word_to_vec_map.items()):
        emb_matrix[i,:] = emb

    # define, build, set weight for keras embedding layer
    embedding_layer = Embedding(vocab_size, emb_dim, trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def build_model_rnn(input_shape, word_to_vec_map):
    """
    # Create a RNN model with LSTM

    Parameters
    ----------
    input_shape : tuple
        The shape of the input, usually (max_len,)
    word_to_vec_map : dict
        Dictionary mapping every word in a vocabulary into its 50-dimensional vector representation

    Returns
    -------
    A model instance in Keras
    """

    # Define sentence_indices as the input of the graph.
    # It should be of shape input_shape and dtype 'int32' (as it contains indices, which are integers).
    input_sentence_indices = Input(shape=input_shape, dtype='int32')

    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map)

    # Propagate sentence_indices through your embedding layer
    # (See additional hints in the instructions).
    embeddings = embedding_layer(input_sentence_indices)  

    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # The returned output should be a batch of sequences, So, set return_sequences = True
    # If return_sequences = False, the LSTM returns only tht last output in output sequence
    X = LSTM(units=128, return_sequences=True)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # The returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(units=128, return_sequences=False)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with 5 units
    X = Dense(5)(X)
    # Add a softmax activation
    X = Activation('softmax')(X)
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=input_sentence_indices, outputs=X)

    return model


def get_synonyms(word):
    """
    Retrieve synonyms for a given word from WordNet.

    Parameters
    ----------
    word : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    synonyms = wordnet.synsets(word)
    synonym_words = set()
    for syn in synonyms:
        for lemma in syn.lemmas():
            synonym_words.add(lemma.name().replace('_', ' '))
    return list(synonym_words)


def synonym_replacement(words, n=2):
    """
    Replace n words in the sentence with synonyms.

    Parameters
    ----------
    words : _type_
        _description_
    n : int, optional
        _description_, by default 2

    Returns
    -------
    _type_
        _description_
    """
    new_words = words[:]
    random_word_list = list(set([word for word in words if get_synonyms(word)]))
    random.shuffle(random_word_list)

    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if synonyms:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return new_words


def random_insertion(words, n=1):
    """
    Randomly insert n words into the sentence.

    Parameters
    ----------
    words : _type_
        _description_
    n : int, optional
        _description_, by default 1

    Returns
    -------
    _type_
        _description_
    """
    new_words = words[:]
    for _ in range(n):
        add_word = random.choice(words)
        insert_synonyms = get_synonyms(add_word)
        if insert_synonyms:
            synonym = random.choice(insert_synonyms)
            insert_position = random.randint(0, len(new_words))
            new_words.insert(insert_position, synonym)
    return new_words


def random_swap(words, n=1):
    """
    Randomly swap the positions of two words in the sentence.

    Parameters
    ----------
    words : _type_
        _description_
    n : int, optional
        _description_, by default 1

    Returns
    -------
    _type_
        _description_
    """
    new_words = words[:]
    if len(new_words) >= 2:
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words


def random_deletion(words, p=0.2):
    """
    Randomly delete words with probability p.

    Parameters
    ----------
    words : _type_
        _description_
    p : float, optional
        _description_, by default 0.2

    Returns
    -------
    _type_
        _description_
    """
    if len(words) == 1:
        return words

    new_words = []
    for word in words:
        if random.uniform(0, 1) > p:
            new_words.append(word)

    if len(new_words) == 0:
        return [random.choice(words)]

    return new_words


def augment_sentence(sentence):
    """
    Apply augmentations to a sentence and return modified versions.

    Parameters
    ----------
    sentence : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    words = sentence.split()
    augmented_sentences = []

    # Apply each augmentation once per sentence
    augmented_sentences.append(" ".join(synonym_replacement(words)))
    augmented_sentences.append(" ".join(random_insertion(words)))
    augmented_sentences.append(" ".join(random_swap(words)))
    augmented_sentences.append(" ".join(random_deletion(words)))

    return augmented_sentences


def clean_history(model_history):

    history_df = pd.DataFrame(model_history.history)

    history_df.rename({'accuracy': 'accuracy_train_last_epoch',
                       'loss': 'loss_train_last_epoch',
                       'val_accuracy': 'accuracy_val_last_epoch',
                       'val_loss': 'loss_val_last_epoch'},
                       axis=1,
                       inplace=True)

    return history_df

