import os 
import json
import math
import pandas as pd
import collections
import logging
import preprocessor as twitter_processor
import twikenizer as twk
from sklearn.model_selection import train_test_split
from tokenizers import BertWordPieceTokenizer
import tensorflow as tf
import numpy as np
seed = 123456789
np.random.seed(seed)
tf.random.set_random_seed(seed)

logging.basicConfig(level=logging.INFO)
twk = twk.Twikenizer()


class CovidDataset:

    def __init__(self, xls_file, export_dir):
        self.filepath = xls_file
        if not os.path.exists(export_dir):
            os.mkdir(export_dir)
        self.export_dir = export_dir
        self.raw_data_file = os.path.join(export_dir, "raw.csv")

        self.raw_train_file = os.path.join(export_dir, "raw.train.csv")
        self.raw_dev_file = os.path.join(export_dir, "raw.dev.csv")
        self.raw_test_file = os.path.join(export_dir, "raw.test.csv")

        self.tok_train_file = os.path.join(export_dir, "tok.train.csv")
        self.tok_dev_file = os.path.join(export_dir, "tok.dev.csv")
        self.tok_test_file = os.path.join(export_dir, "tok.test.csv")

        twitter_processor.set_options(twitter_processor.OPT.URL, twitter_processor.OPT.EMOJI,
                                      twitter_processor.OPT.SMILEY)

        self.datatype_list = ["train", "dev", "test"]

    @staticmethod
    def _read_csv(file):
        df = pd.read_csv(file)
        return list(df["text"]), list(df["label"])

    def _read_excel(self):
        if os.path.isfile(self.raw_data_file):
            # df = pd.read_csv(self.raw_data_file)
            return self._read_csv(self.raw_data_file)

        reader = pd.ExcelFile(self.filepath)

        text_list = []
        label_list = []
        c = 0
        for sheet_name in reader.sheet_names:
            df = reader.parse(sheet_name)
            for d in df.values.tolist():
                c += 1

                if not isinstance(d[2], str):
                    continue

                text = str(d[2]).strip()
                label = str(d[1])
                text = twitter_processor.clean(text)
                if not text:
                    continue

                if label == 5:
                    label = 1
                label_list.append(label)
                text_list.append(text)

        df = pd.DataFrame(data={"text": text_list, "label": label_list})
        df.to_csv(self.raw_data_file)
        logging.info("Data read from excel.")
        return text_list, label_list

    def _train_dev_test_split(self):
        text_list, label_list = self._read_excel()

        for i, t in enumerate(text_list):
            if not isinstance(t, str):
                print(i, t, type(t))
                return

        text_train, text_test, label_train, label_test \
            = train_test_split(text_list, label_list, train_size=0.9, random_state=seed)

        text_train, text_dev, label_train, label_dev \
            = train_test_split(text_train, label_train, train_size=0.9, random_state=seed)

        # text_train_tok = [" ".join(twk.tokenize(t)) for t in text_train]
        # text_dev_tok = [" ".join(twk.tokenize(t)) for t in text_dev]
        # text_test_tok = [" ".join(twk.tokenize(t)) for t in text_test]

        for data_type in self.datatype_list:
            df = pd.DataFrame(data={"text": eval("text_"+data_type), "label": eval("label_"+data_type)})
            df.to_csv(eval("self.raw_%s_file" % data_type))

            df = pd.DataFrame(data={"text": eval("text_" + data_type+"_tok"), "label": eval("label_" + data_type)})
            df.to_csv(eval("self.tok_%s_file" % data_type))

            logging.info("Split %s completed." % data_type)

        return

    def convert_to_fasttext(self, fasttext_dir):
        label_prefix = "__label__"
        if not os.path.exists(fasttext_dir):
            os.mkdir(fasttext_dir)

        for data_type in self.datatype_list:
            text_list, label_list = self._read_csv(eval("self.tok_%s_file" % data_type))
            # Cased and uncased file
            cased_file = os.path.join(fasttext_dir, data_type+".cased.txt")
            uncased_file = os.path.join(fasttext_dir, data_type+".uncased.txt")
            with open(cased_file, "w", encoding="utf-8") as f1, open(uncased_file, "w", encoding="utf-8") as f2:
                for text, label in zip(text_list, label_list):
                    f1.write(
                        "{}{} {}\n".format(label_prefix, label, text)
                    )
                    f2.write(
                        "{}{} {}\n".format(label_prefix, label, text.lower())
                    )
        logging.info("Convert data for fasttext model.")

    def convert_to_rcnn(self, rcnn_dir, do_lower=True):
        if not os.path.exists(rcnn_dir):
            os.mkdir(rcnn_dir)

        for data_type in self.datatype_list:
            text_list, label_list = self._read_csv(eval("self.tok_%s_file" % data_type))
            json_file = os.path.join(rcnn_dir, data_type+".rcnn.json")
            with open(json_file, "w", encoding="utf-8") as fout:
                for text, label in zip(text_list, label_list):
                    if do_lower:
                        text = text.lower()
                    tokens = text.split()
                    label = str(label)
                    example = {
                        "doc_label": [label],
                        "doc_token": tokens,
                        "doc_keyword": [],
                        "doc_topic": []
                    }
                    json_str = json.dumps(example, ensure_ascii=False)
                    fout.write(json_str)
                    fout.write("\n")
        logging.info("Convert data for rcnn model.")

    def convert_to_ratt(self, ratt_dir, do_lower=True, max_sequence_length=128, data_type="train"):
        if not os.path.exists(ratt_dir):
            os.mkdir(ratt_dir)
        # Build dictionary
        text_list, label_list = self._read_csv(self.raw_data_file)

        # Token vocab
        token_vocab_name = "ratt"
        vocab_file = os.path.join(ratt_dir, token_vocab_name+"-vocab.txt")
        if not os.path.isfile(vocab_file):
            tokenizer = BertWordPieceTokenizer(lowercase=do_lower)
            tokenizer.train(files=[self.raw_data_file],
                            vocab_size=8192
                            )
            tokenizer.save_model(ratt_dir, token_vocab_name)
        else:
            tokenizer = BertWordPieceTokenizer(vocab_file=vocab_file, lowercase=do_lower)

        # Label vocab
        label_vocab_file = os.path.join(ratt_dir, "label_dict.txt")
        if not os.path.isfile(label_vocab_file):
            labels = set(label_list)
            label_map = {str(l): i for i, l in enumerate(labels)}
            with open(label_vocab_file, "w", encoding="utf-8") as fout:
                for l in labels:
                    fout.write("%s\n" % l)
        else:
            label_map = {}
            with open(label_vocab_file, encoding="utf-8") as fin:
                for i, line in enumerate(fin):
                    label_map[line.rstrip()] = i

        if data_type not in ["train", "dev", "test"]:
            data_types = ["train", "dev", "test"]
        else:
            data_types = [data_type]

        for data_type in data_types:
            logging.info("Converting %s.." % eval("self.raw_%s_file" % data_type))
            text_list, label_list = self._read_csv(eval("self.raw_%s_file" % data_type))

            outputs = tokenizer.encode_batch(text_list, add_special_tokens=True)
            input_ids = [output.ids for output in outputs]
            padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
                input_ids, padding="post", maxlen=max_sequence_length, truncating="post"
            )

            label_ids = [label_map[str(label)] for label in label_list]
            save_file = os.path.join(ratt_dir, data_type + ".npz")
            np.savez(save_file, inputs=padded_inputs, targets=label_ids)

    def convert_to_ratt_mlm(self, mlm_dir, do_lower=True, max_sequence_length=128, data_type="train"):
        logging.info("Convert data from mlm.")
        self.raw_train_file = os.path.join(self.export_dir, "raw.train.mlm.csv")
        self.convert_to_ratt(mlm_dir, do_lower, max_sequence_length, data_type)
        logging.info("Data converting to %s" % mlm_dir)
        # Change back to original
        self.raw_train_file = os.path.join(self.export_dir, "raw.train.csv")

    def data_augmentation_mlm(self, mlm_file):
        """Data augmentation for training set"""
        # Convert to file suitable for https://github.com/jasonwei20/eda_nlp
        train_texts, train_labels = self._read_csv(self.raw_train_file)
        mlm_train_texts, mlm_train_labels = self._read_csv(mlm_file)

        label_counts = collections.Counter(mlm_train_labels)
        _, min_size = label_counts.most_common()[-1]
        c = 0
        for t, l in zip(mlm_train_texts, mlm_train_labels):
            if l == 4:
                continue
                # c += 1
                # if c >= min_size:
                #     continue
            train_texts += [t]
            train_labels += [l]

        # df = pd.DataFrame(data={
        #     "text": train_texts + mlm_train_texts,
        #     "label": train_labels + mlm_train_labels
        # })
        df = pd.DataFrame(data={
            "text": train_texts,
            "label": train_labels
        })
        csv_file = os.path.join(self.export_dir, "raw.train.mlm.csv")
        df.to_csv(csv_file)
        logging.info("mlm csv file saved to %s" % csv_file)


def main():

    xls_file = "data/covid-19.xlsx"
    export_dir = "data/train_data"
    fasttext_dir = os.path.join(export_dir, "fasttext")
    rcnn_dir = os.path.join(export_dir, "rcnn")
    ratt_dir = os.path.join(export_dir, "ratt")
    ratt_mlm_dir = os.path.join(export_dir, "ratt_mlm")
    mlm_file= os.path.join(export_dir, "masked.train.csv")

    dataset = CovidDataset(xls_file, export_dir)

    # ret = dataset._read_excel()
    # ret = dataset._train_dev_test_split()

    dataset.convert_to_fasttext(fasttext_dir)
    dataset.convert_to_rcnn(rcnn_dir)
    dataset.convert_to_ratt(ratt_dir, data_type="all", max_sequence_length=128)

    dataset.data_augmentation_mlm(mlm_file)
    dataset.convert_to_ratt_mlm(ratt_mlm_dir, data_type="all", max_sequence_length=128)


if __name__ == '__main__':
    main()
