import time

import numpy as np
import tensorflow as tf

from audio_reader import AudioReader
from constants import c
from file_logger import FileLogger
from utils import FIRST_INDEX

# Some configs
num_features = 13
# Accounting the 0th index +  space + blank label = 28 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1

# Hyper-parameters
num_epochs = 500000
num_hidden = 100
num_layers = 1
batch_size = 1

num_examples = 1
num_batches_per_epoch = int(num_examples / batch_size)

audio = AudioReader(audio_dir=c.AUDIO.VCTK_CORPUS_PATH,
                    sample_rate=c.AUDIO.SAMPLE_RATE)

file_logger = FileLogger('out.tsv', ['curr_epoch',
                                     'train_cost',
                                     'train_ler',
                                     'val_cost',
                                     'val_ler',
                                     'random_shift'])


def run_ctc():

    def next_training_batch():
        import random
        from utils import convert_inputs_to_ctc_format
        # random_index = random.choice(list(audio.cache.keys()))
        # random_index = list(audio.cache.keys())[0]
        random_index = random.choice(list(audio.cache.keys())[0:2942])
        # print("next_training_batch" + random_index)
        training_element = audio.cache[random_index]
        target_text = training_element['target']
        train_inputs, train_targets, train_seq_len, original = convert_inputs_to_ctc_format(training_element['audio'],
                                                                                            c.AUDIO.SAMPLE_RATE,
                                                                                            target_text)
        return train_inputs, train_targets, train_seq_len, original

    def next_testing_batch():
        import random
        from utils import convert_inputs_to_ctc_format
        random_index = random.choice(list(audio.cache.keys())[0:2942])
        # print("next_testing_batch" + random_index)
        training_element = audio.cache[random_index]
        target_text = training_element['target']
        random_shift = np.random.randint(low=1, high=2942)
        # print('random_shift =', random_shift)
        # truncated_audio = training_element['audio'][random_shift:]
        truncated_audio = training_element['audio']
        train_inputs, train_targets, train_seq_len, original = convert_inputs_to_ctc_format(truncated_audio,
                                                                                            c.AUDIO.SAMPLE_RATE,
                                                                                            target_text)
        return train_inputs, train_targets, train_seq_len, original, random_shift

    with tf.Session() as session:
        saver = tf.train.import_meta_graph('model\my_ctc_model-100.meta')
        saver.restore(session, tf.train.latest_checkpoint('model\\'))

        # Now, let's access and create placeholders variables and
        # create feed-dict to feed new data

        graph = tf.get_default_graph()

        inputs = graph.get_tensor_by_name("input_features:0")
        seq_len = graph.get_tensor_by_name("input_seq_len:0")

        output_0 = graph.get_tensor_by_name("output/indices:0")
        output_1 = graph.get_tensor_by_name("output/values:0")
        output_2 = graph.get_tensor_by_name("output/shape:0")

        targets = tf.SparseTensor(indices=output_0, values=output_1, dense_shape=output_2)

        decoded_0 = graph.get_tensor_by_name("CTCGreedyDecoder:0")
        decoded_1 = graph.get_tensor_by_name("CTCGreedyDecoder:1")
        decoded_2 = graph.get_tensor_by_name("CTCGreedyDecoder:2")

        decode = tf.SparseTensor(indices=decoded_0, values=decoded_1, dense_shape=decoded_2)

        cost = graph.get_tensor_by_name("cost:0")
        optimizer = graph.get_operation_by_name("optimizer")
        ler = graph.get_tensor_by_name("ler:0")

        for curr_epoch in range(num_epochs):
            train_cost = train_ler = 0
            start = time.time()

            for batch in range(num_batches_per_epoch):
                train_inputs, train_targets, train_seq_len, original = next_training_batch()
                feed = {inputs: train_inputs,
                        targets: train_targets,
                        seq_len: train_seq_len}

                batch_cost, _ = session.run([cost, optimizer], feed)
                train_cost += batch_cost * batch_size
                train_ler += session.run(ler, feed_dict=feed) * batch_size

            train_cost /= num_examples
            train_ler /= num_examples

            val_inputs, val_targets, val_seq_len, val_original, random_shift = next_testing_batch()
            val_feed = {inputs: val_inputs,
                        targets: val_targets,
                        seq_len: val_seq_len}

            val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)

            # # Decoding
            d = session.run(decode, feed_dict=val_feed)
            str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
            # Replacing blank label to none
            str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
            # Replacing space label to space
            str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')

            print('Original val: %s' % val_original)
            print('Decoded val: %s' % str_decoded)

            log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, " \
                  "val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"

            file_logger.write([curr_epoch + 1,
                               train_cost,
                               train_ler,
                               val_cost,
                               val_ler,
                               random_shift])

            print(log.format(curr_epoch + 1, num_epochs, train_cost, train_ler,
                             val_cost, val_ler, time.time() - start))
            # saver = tf.train.Saver()
            #
            # # sess.run(tf.global_variables_initializer())
            # saver.save(session, 'model\model.ckpt')
            # saver = tf.train.Saver()

            # sess.run(tf.global_variables_initializer())
            # saver.save(session, 'model\model.ckpt')
            # saver.save(session, 'model\my_ctc_model', global_step=None)

        # Decoding
        d = session.run(decode, feed_dict=val_feed)
        str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
        # Replacing blank label to none
        str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
        # Replacing space label to space
        str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')

        print('Original val: %s' % val_original)
        print('Decoded val: %s' % str_decoded)

        # inputs = graph.get_tensor_by_name("input_features:0")
        # seq_len = graph.get_tensor_by_name("input_seq_len:0")

        print(inputs)
        print(seq_len)
        # print(logits)
        # saver = tf.train.Saver()

        # sess.run(tf.global_variables_initializer())
        # saver.save(session, 'model\model.ckpt')
        saver.save(session, 'model\my_ctc_model', global_step=100)


if __name__ == '__main__':
    run_ctc()
