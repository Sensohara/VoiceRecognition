import numpy as np
import tensorflow as tf
from constants import c
from utils import FIRST_INDEX
from text_helper import convert_telex_to_unicode

model_file = 'model\model.ckpt'
test_set = 'TEST_SET\\'
validation_set = "VALIDATION_SET\\"
target_path_audio = r"D://LUAN AN TOT NGHIEP//phong_bui-vnrecognizectctrain-New//phong_bui-vnrecognizectctrain-7399eb2125b0//Volumes//Transcend//VCTK-Corpus//vctk-p225//wav48//p225//"
test_folder = target_path_audio


def run_ctc():
    def next_testing_file(file_name):
        from audio_reader import read_audio_from_filename
        truncated_audio = read_audio_from_filename(file_name, sample_rate=c.AUDIO.SAMPLE_RATE)
        from utils import convert_inputs_to_ctc_format
        test_inputs, targets, test_seq_len, original = convert_inputs_to_ctc_format(truncated_audio,
                                                                                    c.AUDIO.SAMPLE_RATE,
                                                                                    "")
        return test_inputs, test_seq_len, original

    with tf.Session() as session:
        saver = tf.train.import_meta_graph('model\my_ctc_model-100.meta')
        saver.restore(session, tf.train.latest_checkpoint('model\\'))

        # Now, let's access and create placeholders variables and
        # create feed-dict to feed new data

        graph = tf.get_default_graph()

        inputs = graph.get_tensor_by_name("input_features:0")
        seq_len = graph.get_tensor_by_name("input_seq_len:0")

        decoded_0 = graph.get_tensor_by_name("CTCGreedyDecoder:0")
        decoded_1 = graph.get_tensor_by_name("CTCGreedyDecoder:1")
        decoded_2 = graph.get_tensor_by_name("CTCGreedyDecoder:2")

        decode = tf.SparseTensor(indices=decoded_0, values=decoded_1, dense_shape=decoded_2)

        print(decoded_0)
        print(decoded_1)
        print(decoded_2)
        print(decode)

        print(inputs)
        print(seq_len)

        import os
        files = os.listdir(test_folder)
        for file in files:
            file_name = test_folder + file
            val_inputs, val_seq_len, val_original = next_testing_file(file_name)
            feed = {
                        inputs: val_inputs,
                        seq_len: val_seq_len
                    }
            # Decoding
            d = session.run(decode, feed_dict=feed)
            str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
            # Replacing blank label to none
            str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
            # Replacing space label to space
            str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')

            print('Original val: %s' % convert_telex_to_unicode(val_original))
            print('Decoded val: %s' % convert_telex_to_unicode(str_decoded))
            print('\n-------------------------------------------------')


if __name__ == '__main__':
    run_ctc()