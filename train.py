import os
import collections
import math
import random
import tensorflow as tf
import numpy as np
# import sklearn.metrics
# import glob
# from absl import flags, app
import logging
logging.basicConfig(level=logging.INFO)

try:
    from model import TextRAtt
except:
    from covid_classification.model import TextRAtt


seed = 123456789
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

config = {
    "learning_rate": 0.005,
    "use_class_weight": False,
    "batch_size": 16,
    "epochs": 5,
    "train_dir": "data/train_data/ratt_mlm",
    "do_train": True,
    "do_predict": True,
    "model_dir": "/path/to/model/ratt_mlm",
    "result_dir": "/path/to/result/ratt_mlm",
    "model_name": "ratt_mlm"
}


def load_data(filepath):
    np_data = np.load(filepath, allow_pickle=True)
    return np_data["inputs"], np_data["targets"]


def make_model(config):
    model = TextRAtt()
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.get("learning_rate", 0.005))
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=["acc"])
    return model


def train_model(config):

    train_file = os.path.join(config.get("train_dir"), "train.npz")
    val_file = os.path.join(config.get("train_dir"), "dev.npz")
    test_file = os.path.join(config.get("train_dir"), "test.npz")
    model_path = os.path.join(config.get("model_dir"), "model")

    if config.get("do_train", False):
        logging.info("Training..")

        trainset = load_data(train_file)
        valset = load_data(val_file)

        x_train, y_train = trainset
        x_val, y_val = valset
        # x_test, y_test = testset

        model = make_model(config)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode="min", verbose=1)
        checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                         monitor="val_acc", mode="max", verbose=1, save_weights_only=True,
                                                         save_best_only=True)
        callbacks = [early_stopping, checkpoints]
        # if config.get("ensemble_training", False):
        #     callbacks = None

        class_weight = None
        if config.get("use_class_weight", False):
            label_count = collections.Counter(y_train.tolist())
            num_classes = len(label_count)
            num_sample = y_train.size
            class_weight = {}
            for l, c in label_count.items():
                class_weight[l] = math.log(num_sample/c) + 1
                # class_weight[l] = 1 / c * num_sample / num_classes
                # class_weight[l] = num_sample / c

        model.fit(
            x_train, y_train,
            batch_size=config.get("batch_size", 16),
            epochs=config.get("epochs", 5),
            validation_data=(x_val, y_val),
            validation_batch_size=16,
            class_weight=class_weight,
            callbacks=callbacks,
        )

    if config.get("do_predict", False):
        logging.info("Predicting..")

        x_test, y_test = load_data(test_file)
        model = make_model(config)
        model.build(input_shape=(16, 128))
        model.load_weights(model_path).expect_partial()

        outputs = model.predict(x_test)
        predictions = tf.argmax(outputs, axis=1).numpy()
        np.savez(os.path.join(config.get("result_dir"), "%s.npz" % config.get("model_name")),
                 prediction=predictions, label=y_test)
    return True

def ensemble_train_and_predict(config,
                               epochs=3,
                               num_average=3):

    train_file = os.path.join(config.get("train_dir"), "train.npz")
    val_file = os.path.join(config.get("train_dir"), "dev.npz")
    test_file = os.path.join(config.get("train_dir"), "test.npz")
    model_dir = config.get("model_dir")

    x_train, y_train = load_data(train_file)
    valset = load_data(val_file)
    x_test, y_test = load_data(test_file)

    model = make_model(config)
    model_path = os.path.join(model_dir, "model_average", "ckpt-{epoch:04d}.ckpt")
    checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                     monitor="val_acc", mode="max", verbose=1, save_weights_only=True,
                                                     save_best_only=False, save_freq="epoch")

    model.fit(
        x_train, y_train,
        batch_size=16,
        epochs=epochs,
        validation_data=valset,
        validation_batch_size=16,
        callbacks=[checkpoints]
    )

    weights = []
    for i in range(1, num_average+1):
        checkpoint = os.path.join(model_dir, "model_average", "ckpt-{epoch:04d}.ckpt".format(epoch=i))
        model.load_weights(checkpoint).expect_partial()
        # models.append(model)
        model.build(input_shape=(16, 128))
        weight = model.get_weights()
        # print(type(weight))
        weights.append(weight)

    # weights = [model.get_weights() for model in models]
    # print(weights)
    new_weights = list()
    for weights_list_tuple in zip(*weights):
        new_weights.append(
            np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
        )

    model.set_weights(new_weights)

    # Prediction
    outputs = model.predict(x_test)
    predictions = tf.argmax(outputs, axis=1).numpy()
    np.savez(os.path.join(config.get("result_dir"), "%s_ensemble.npz" % config.get("model_name")),
             prediction=predictions, label=y_test)


if __name__ == '__main__':
    train_model(config)
    # ensemble_train_and_predict(config, epochs=5, num_average=3)
