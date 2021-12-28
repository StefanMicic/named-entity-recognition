import os.path
from collections import Counter

import numpy as np
import tensorflow as tf
from datasets import load_dataset
from tensorflow import keras

from models.custom_loss import CustomNonPaddingTokenLoss
from models.ner_model import NERModel
from utils import calculate_metrics, export_to_file, lowercase_and_convert_to_ids, make_tag_lookup_table, \
    map_record_to_training_data

if __name__ == "__main__":
    conll_data = load_dataset("conll2003")

    if not os.path.exists("data"):
        os.mkdir("data")
        export_to_file("./data/conll_train.txt", conll_data["train"])
        export_to_file("./data/conll_val.txt", conll_data["validation"])

    mapping = make_tag_lookup_table()
    print(mapping)

    all_tokens = sum(conll_data["train"]["tokens"], [])
    all_tokens_array = np.array(list(map(str.lower, all_tokens)))

    counter = Counter(all_tokens_array)
    print(len(counter))

    num_tags = len(mapping)
    vocab_size = 20000

    vocabulary = [token for token, count in counter.most_common(vocab_size - 2)]

    lookup_layer = keras.layers.StringLookup(
        vocabulary=vocabulary
    )

    train_data = tf.data.TextLineDataset("./data/conll_train.txt")
    val_data = tf.data.TextLineDataset("./data/conll_val.txt")

    batch_size = 32
    train_dataset = (
        train_data.map(map_record_to_training_data)
            .map(lambda x, y: (lowercase_and_convert_to_ids(x, lookup_layer), y))
            .padded_batch(batch_size)
    )
    val_dataset = (
        val_data.map(map_record_to_training_data)
            .map(lambda x, y: (lowercase_and_convert_to_ids(x, lookup_layer), y))
            .padded_batch(batch_size)
    )

    ner_model = NERModel(num_tags, vocab_size, embed_dim=32, num_heads=4, ff_dim=64)

    loss = CustomNonPaddingTokenLoss()

    ner_model.compile(optimizer="adam", loss=loss)

    ner_model.fit(train_dataset, epochs=2)


    def tokenize_and_convert_to_ids(text):
        tokens = text.split()
        return lowercase_and_convert_to_ids(tokens, lookup_layer)


    # Sample inference using the trained model
    sample_input = tokenize_and_convert_to_ids(
        "eu rejects german call to boycott british lamb"
    )
    sample_input = tf.reshape(sample_input, shape=[1, -1])
    print(sample_input)

    output = ner_model.predict(sample_input)
    ner_model.summary()
    print(output.shape)
    prediction = np.argmax(output, axis=-1)[0]
    prediction = [mapping[i] for i in prediction]

    # eu -> B-ORG, german -> B-MISC, british -> B-MISC
    print(prediction)

    calculate_metrics(val_dataset, ner_model, mapping)
