#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

# Parameters
# ==================================================

# Data Parameters
datafolder="C:\\Users\\OSU user\\Desktop\\share\\stuff\\OutData\\Collusion\\"
tf.flags.DEFINE_string("positive_data_file", datafolder+"truth", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file",  datafolder+"biased", "Data source for the negative data.")
tf.flags.DEFINE_string("neutral_data_file",  datafolder+"neutral", "Data source for the neutral data.")

tf.flags.DEFINE_boolean("isSingleData",  True, "Set a single data source to test")
tf.flags.DEFINE_string("singleDataFile",  datafolder+"collusion_data", "Data source i you have single file")
tf.flags.DEFINE_list("patternToencode",  [1,0,0], "pattern to represent above file")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "C:\\Users\\OSU user\\PycharmProjects\\CNN_Classifier\\runs\\1562613609\\checkpoints\\", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS._flags().items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    if FLAGS.isSingleData:
        x_textlist, x, y=data_helpers.load_and_predict_labels(FLAGS.singleDataFile,80,FLAGS.patternToencode)
        y_test = np.argmax(y, axis=1)
        x_raw = [" ".join(x).replace("\n", "") for x in x_textlist]

    else:
        x_text,x_train, y_train,x_test,y_test, vocabulary, vocabulary_inv,vocabobj,x_textlist = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file,FLAGS.neutral_data_file,80)
        y_test = np.argmax(np.vstack((y_train,y_test)), axis=1)
        x_raw=[" ".join(x).replace("\n","") for x in x_textlist]
else:
    x_raw = ["what the fuck is wrong with you , you racist scum and trump is responsible .i hate him ","a masterpiece four years in the making an astute good look", "an amazing thing to do"]
    y_test = [[1,0,0], [0,0,1],[0,1,0]]
    y_test = np.argmax(y_test, axis=1)

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw))).astype('float32')

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob:1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.tsv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f,delimiter='\t').writerows(predictions_human_readable)
