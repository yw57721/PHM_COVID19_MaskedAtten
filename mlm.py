import spacy
import re
import os
import pandas as pd
from flashtext.keyword import KeywordProcessor
from fairseq.models.roberta import RobertaModel
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq import options


nlp = spacy.load("en_core_web_sm")
ignore_word_regex = re.compile(r"(coronavirus|covid|corona|my|test|positive|negative|virus)")
keyword_processor = KeywordProcessor(case_sensitive=True)

def find_mask_word_per_tweet(text):
    words = []
    for doc in nlp(text):
        if doc.is_stop:
            continue
        if len(doc.text) <= 1:
            continue
        if re.findall(ignore_word_regex, doc.text):
            continue
        if doc.pos_ in ["PROPN", "VERB", "NOUN"]:
            words.append(doc.text)
            # print(doc.text, doc.pos_)
    return words


def _init_model(pretrain_model):
    bpe_path = os.path.join(pretrain_model, "bpe.codes")

    BERTweet = RobertaModel.from_pretrained(pretrain_model, checkpoint_file='model.pt')
    BERTweet.eval()  # disable dropout (or leave in train mode to finetune)

    # Incorporate the BPE encoder into BERTweet-base

    parser = options.get_preprocessing_parser()
    parser.add_argument('--bpe-codes', type=str, help='path to fastBPE BPE',
                        default=bpe_path)
    args = parser.parse_args()
    BERTweet.bpe = fastBPE(args)  # Incorporate the BPE encoder into BERTweet
    return BERTweet


def get_masked_sentences(pretrain_model, train_file, masked_train_file):
    df = pd.read_csv(train_file)
    bert_tweet = _init_model(pretrain_model)

    extra_text = []
    extra_label = []
    i = 0
    for text, label in zip(df["text"], df["label"]):
        if i % 100 == 0:
            print("processing %s sentence.." % i)
        i += 1
        if label == 3:
            continue
        words = find_mask_word_per_tweet(text)
        # line = "\t".join(words)
        # fout.write(line+"\n")
        text_copy = text
        for word in words:
            masked_text = re.sub(r"\b%s" % re.escape(word), " <mask>", text, count=1, flags=1)
            try:
                topk_filled_outputs = bert_tweet.fill_mask(masked_text, topk=5)
            except Exception as e:
                print(masked_text)
                continue
            for candidate in topk_filled_outputs:
                candidate_text, _, repl = candidate
                if candidate_text.lower() != text.lower():
                    # extra_text.append(candidate_text)
                    # extra_label.append(label)
                    # break
                    text_copy = re.sub(r"\b%s" % re.escape(word), repl, text_copy)
        if text_copy != text:
            extra_text.append(text)
            extra_label.append(label)

        df = pd.DataFrame(data={
            "text": extra_text,
            "label": extra_label
        }
        )
        df.to_csv(masked_train_file)
    #
    print("done")

    

if __name__ == '__main__':
    pretrained_model = "/path/to/pretrained_model"
    train_file = "data/raw.train.csv"
    masked_train_file = "data/masked.train.csv"
    get_masked_sentences(pretrained_model, train_file, masked_train_file)
