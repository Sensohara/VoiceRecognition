from pydub import AudioSegment
from pydub.playback import play
from pydub.silence import split_on_silence
import os
import shutil
import Levenshtein
import numpy as np
import tensorflow as tf
from constants import c
from utils import FIRST_INDEX
from text_helper import convert_telex_to_unicode


source_path = r"C://Users//SIMON//Documents//Sound recordings//"

target_path_txt = r"D://LUAN AN TOT NGHIEP//phong_bui-vnrecognizectctrain-New//phong_bui-vnrecognizectctrain-7399eb2125b0//Volumes//Transcend//VCTK-Corpus//vctk-p225//txt//p225"
target_path_audio = r"D://LUAN AN TOT NGHIEP//phong_bui-vnrecognizectctrain-New//phong_bui-vnrecognizectctrain-7399eb2125b0//Volumes//Transcend//VCTK-Corpus//vctk-p225//wav48//p225"


def create_text_file():
    a = []
    b = []
    i = 1
    filename_replace = ""
    for root, dirs, filenames in os.walk(source_path):
        if i == 1:
            a = dirs
        if i != 1:
            b.append(root)
        i += 1

    for i, j in zip(a, b):
        # interator de tao text file name
        x = 0
        for filename in os.listdir(j):
            x += 1
            # tao file text theo tu muon train
            txt_path = j + "//" + i + "({0})".format(x) + ".txt"
            f = open(txt_path, "w+")
            f.write(i)

            f.close()

            # di chuyen file text toi folder source code
            shutil.move(txt_path, txt_path.replace(j, target_path_txt))

            # doi ten sao cho giong voi file text, filename input phai bat dau la "Recording "
            filename = j + "//" + filename
            filename_replace = j + "//" + i + "({0})".format(x) + ".m4a"
            shutil.move(filename, filename_replace)  # doi ten file

            filename = filename_replace
            filename_replace = filename.replace(j, target_path_audio)
            shutil.move(filename, filename_replace)  # di chuyen den folder source code

            # convert file type to wav
            sound_file = AudioSegment.from_file(filename_replace)
            print(filename_replace)
            sound_file.export(filename_replace.replace("m4a", "wav"), format="wav")

            # xoa file cu
            os.remove(filename_replace)
# =============================================================================================================================

#==============================================================================================================================
# split_file = r"D://LUAN AN TOT NGHIEP//phong_bui-vnrecognizectctrain-New//phong_bui-vnrecognizectctrain-7399eb2125b0//splitAudio"
# def split_audio_1():
#     for filename in os.listdir(target_path_audio):
#         sound_file_0 = AudioSegment.from_wav(target_path_audio + "//" + filename)
#         audio_chunks = split_on_silence(sound_file_0, min_silence_len=100,  silence_thresh=-50)
#         i = 1
#         for chunk in audio_chunks:
#             print("exporting",chunk)
#             file = filename.replace( ".wav", "_{0}.wav".format(i))
#             postfix_0 = split_file + "//" + file
#             print(postfix_0)
#             chunk.export(postfix_0, format="wav")
#             i += 1


#     # for filename in os.listdir(split_file):
#     #     a = split_file + "//" + filename
#     #     statinfo = os.stat(a)
#     #     if statinfo.st_size < 102400:
#     #         os.remove(a)

#     for filename in os.listdir(split_file):
#         a = split_file + "//" + filename
#         shutil.move(a, a.replace("_1", "").replace("_2", "").replace("_3", ""))
# # ======================================================================================================================
# # percent = SequenceMatcher(None, 'gui cho thay tho the nhan su', 'guiaa cho thayf th nhan su').ratio()
# # percent_1 = Levenshtein.ratio('gui cho thay tho the nhan su', 'guiaa cho thayf th nhan su')
# #
# # print(percent_1)
# # ======================================================================================================================
# # split_audio()
# # create_text_file()
# #==============================================================================================================================


def split_audio():
    test_file = r"D://LUAN AN TOT NGHIEP//phong_bui-vnrecognizectctrain-New//phong_bui-vnrecognizectctrain-7399eb2125b0//Source audio"
    source_path = r"C://Users//SIMON//Documents//Sound recordings//"
    for filename in os.listdir(source_path):
        filename = source_path + '//' + filename
        shutil.move(filename, filename.replace(source_path, test_file))

    for filename in os.listdir(test_file):
        filename = test_file + '//' + filename
        sound_file = AudioSegment.from_file(filename)
        sound_file.export(filename.replace("m4a","wav"), format="wav")
        if "wav" not in filename:
            os.remove(filename)
        print(filename)
    j = 1
    for filename in os.listdir(test_file):
        filename = test_file + '//' + filename
        sound_file = AudioSegment.from_wav(filename)
        max = 0
        final_chunk = []
        for i in range(-35,-10):
            audio_chunks = split_on_silence(sound_file, min_silence_len=100,  silence_thresh=i)
            print(str(i) + ' ' + str(len(audio_chunks)) )
            if len(audio_chunks) > max:
                max = len(audio_chunks)
                final_chunk = audio_chunks

        i = 1
        for chunk in final_chunk:
            postfix_0 = "//sc{0}_{1}.wav"
            postfix_0 = test_file.replace("//Source audio", "//TEST_SET") + postfix_0
            print(postfix_0.format(j,i))
            chunk.export(postfix_0.format(j,i), format="wav")
            i += 1
        j += 1


#==============================================================================================================================
test_folder = 'TEST_SET\\'
def run_ctc_customize():
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
        count = 1
        result = ''
        print('\n')
        for file in files:
            if int(file[2]) == count + 1:
                print('%02d.Before edit distance: %s' % (count,result))
                result = matching(result)
                print(str(count) + '.Sentence: %s' % convert_telex_to_unicode(result).strip() + '\n')
                count += 1
                result = ''

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

            # print('Original val: %s' % convert_telex_to_unicode(val_original))
            # print('Decoded val: %s' % convert_telex_to_unicode(str_decoded))
            # print('\n-------------------------------------------------')
            result = result + ' ' + str_decoded

        print('%02d.Before edit distance: %s' % (count,result))
        # result = matching(result)
        # print(str(count) + '.Sentence: %s' % convert_telex_to_unicode(result).strip())
        # count += 1
        # result = ''


combination_path = r"D://LUAN AN TOT NGHIEP//phong_bui-vnrecognizectctrain-New//phong_bui-vnrecognizectctrain-7399eb2125b0//combination.txt"

def matching(test_sent):
    f = open(combination_path, "r")
    sentences = f.readlines()
    f.close()
    max = 0
    sent_highest_rate = ''
    for sentence in sentences:
        if Levenshtein.ratio( test_sent, sentence ) > max:
            max = Levenshtein.ratio( test_sent, sentence )
            sent_highest_rate = sentence
    print('  Rate: %.2f' % max)
    return sent_highest_rate


split_audio()
run_ctc_customize()