import bert
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


class LaBSE:
    def __init__(self, max_seq_length):
        self.model = None
        self.layer = None
        self.tokenizer = None
        self.max_seq_length = max_seq_length

    def set_model(self, path):
        # Load model and layer
        self.model, self.layer = get_model(path, max_seq_length=self.max_seq_length)

        # Bert setting
        vocab_file = self.layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = self.layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)

    def encode(self, input_strings):
        input_ids_all, input_mask_all, segment_ids_all = [], [], []
        for input_string in input_strings:
            # Tokenize input.
            input_tokens = ["[CLS]"] + self.tokenizer.tokenize(input_string) + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            sequence_length = min(len(input_ids), self.max_seq_length)

            # Padding or truncation.
            if len(input_ids) >= self.max_seq_length:
                input_ids = input_ids[:self.max_seq_length]
            else:
                input_ids = input_ids + [0] * (self.max_seq_length - len(input_ids))

            input_mask = [1] * sequence_length + [0] * (self.max_seq_length - sequence_length)

            input_ids_all.append(input_ids)
            input_mask_all.append(input_mask)
            segment_ids_all.append([0] * self.max_seq_length)

        result = self.model([
            np.array(input_ids_all),
            np.array(input_mask_all),
            np.array(segment_ids_all)]).numpy()

        return result


def get_model(model_url, max_seq_length):
    labse_layer = hub.KerasLayer(model_url, trainable=True)

    # Define input.
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                        name="segment_ids")

    # LaBSE layer.
    pooled_output, _ = labse_layer([input_word_ids, input_mask, segment_ids])

    # The embedding is l2 normalized.
    pooled_output = tf.keras.layers.Lambda(
        lambda x: tf.nn.l2_normalize(x, axis=1))(pooled_output)

    # Define model.
    return tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=pooled_output), labse_layer
