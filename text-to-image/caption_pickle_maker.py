
import os
import re
import time
import nltk
import re
import string
import tensorlayer as tl
from utils import *


dataset = '102flowers' #
need_256 = True # set to True for stackGAN



if dataset == '102flowers':
    """
    images.shape = [8000, 64, 64, 3]
    captions_ids = [80000, any]
    """
    cwd = os.getcwd()
    img_dir = os.path.join(cwd, '102flowers/102flowers')
    caption_dir = os.path.join(cwd, '102flowers/text_c10')
    VOC_FIR = cwd + '/vocab.txt'

    ## load captions
    caption_sub_dir = load_folder_list( caption_dir )
    captions_dict = {}
    processed_capts = []
    for sub_dir in caption_sub_dir: # get caption file list
        with tl.ops.suppress_stdout():
            files = tl.files.load_file_list(path=sub_dir, regx='^image_[0-9]+\.txt')
            for i, f in enumerate(files):
                file_dir = os.path.join(sub_dir, f)
                key = int(re.findall('\d+', f)[0])
                t = open(file_dir,'r')
                lines = []
                for line in t:
                    line = preprocess_caption(line)
                    lines.append(line)
                    processed_capts.append(tl.nlp.process_sentence(line, start_word="<S>", end_word="</S>"))
                assert len(lines) == 10, "Every flower image have 10 captions"
                captions_dict[key] = lines
    print(" * %d x %d captions found " % (len(captions_dict), len(lines)))

    ## build vocab
    if not os.path.isfile('vocab.txt'):
        _ = tl.nlp.create_vocab(processed_capts, word_counts_output_file=VOC_FIR, min_word_count=1)
    else:
        print("WARNING: vocab.txt already exists")
    vocab = tl.nlp.Vocabulary(VOC_FIR, start_word="<S>", end_word="</S>", unk_word="<UNK>")

    ## store all captions ids in list
    captions_ids = {}
    try: # python3
        tmp = captions_dict.items()
    except: # python3
        tmp = captions_dict.iteritems()

    for key, value in tmp:
        captions = []
        for v in value:
            captions.append( [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(v)] + [vocab.end_id])
        captions_ids[key] = np.asarray(captions) 
    # captions_ids = np.asarray(captions_ids)
    # print(" * tokenized %d captions" % len(captions_ids))

    ## check
    img_capt = captions_dict[1][1]
    print("img_capt: %s" % img_capt)
    print("nltk.tokenize.word_tokenize(img_capt): %s" % nltk.tokenize.word_tokenize(img_capt))
    img_capt_ids = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(img_capt)]#img_capt.split(' ')]
    print("img_capt_ids: %s" % img_capt_ids)
    print("id_to_word: %s" % [vocab.id_to_word(id) for id in img_capt_ids])


import pickle
def save_all(targets, file):
    with open(file, 'wb') as f:
        pickle.dump(targets, f)

save_all(vocab, '_vocab_t2i.pickle')
save_all(captions_ids, '_caption_t2i.pickle')
