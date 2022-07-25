from keras import Sequential
from keras.layers import LSTM, Dropout, Dense, Embedding, Masking, Bidirectional


def lstm(classes_num, x_train, y_train):

    '''
    # n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
    n_features = len(x_train[0][0])
    model = Sequential()
    # model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(LSTM(100, input_shape=(None, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    # model.add(Dense(n_outputs, activation='softmax'))
    model.add(Dense(100, activation='softmax'))
    '''

    model = Sequential()
    # max_sequence_length = x_train.shape[1]
    # embedding_output_dims = 15
    # num_distinct_words = 5000
    # model.add(Embedding(num_distinct_words, embedding_output_dims, mask_zero=True, input_length=max_sequence_length))
    model.add(Masking(0.0))
    model.add(LSTM(100))
    # model.add(Bidirectional(LSTM(10), merge_mode='sum'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(classes_num, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
