"""Contains a main function for training and/or evaluating a model."""

import os
import sys

import numpy as np
import random
import subprocess

from EditSQL.parse_args import interpret_args
from EditSQL.postprocess_eval import read_schema, read_prediction, postprocess, write_and_evaluate
from EditSQL.eval_scripts.evaluation import build_foreign_key_map_from_json, \
    evaluate as eval_script_evaluate
from EditSQL.data_util import atis_data
from EditSQL.model.schema_interaction_model import SchemaInteractionATISModel
from EditSQL.logger import Logger
from EditSQL.model_util import Metrics, evaluate_utterance_sample, evaluate_interaction_sample, \
    train_epoch_with_utterances, train_epoch_with_interactions, evaluate_using_predicted_queries

import torch
import pdb

np.random.seed(0)
random.seed(0)

VALID_EVAL_METRICS = [Metrics.LOSS, Metrics.TOKEN_ACCURACY, Metrics.STRING_ACCURACY]
FINAL_EVAL_METRICS = [Metrics.STRING_ACCURACY, Metrics.TOKEN_ACCURACY]


def evaluate(model, data, params, last_save_file, split, full_name=None):
    """Evaluates a pretrained model on a dataset.

    Inputs:
        model (ATISModel): Model class.
        data (ATISData): All of the data.
        params (namespace): Parameters for the model.
        last_save_file (str): Location where the model save file is.
    """

    if "data_clean" in params.raw_train_filename:
        raw_data_directory = "EditSQL/data_clean/"
    else:
        raw_data_directory = "EditSQL/data/"

    table_schema_path = os.path.join(raw_data_directory, "spider", "tables.json")
    gold_path = os.path.join(raw_data_directory, "spider", "dev_gold.sql")
    db_path = os.path.join(raw_data_directory, "database/")

    db_list = []
    with open(gold_path) as f:
        for line in f:
            line_split = line.strip().split('\t')
            if len(line_split) != 2:
                continue
            db = line.strip().split('\t')[1]
            if db not in db_list:
                db_list.append(db)

    kmaps = build_foreign_key_map_from_json(table_schema_path)

    if last_save_file:
        model.load(last_save_file)
    else:
        if not params.save_file:
            raise ValueError(
                "Must provide a save file name if not training first.")
        model.load(params.save_file)

    filename = split

    if filename == 'dev':
        split = data.dev_data
    elif filename == 'train':
        split = data.train_data
    elif filename == 'test':
        split = data.test_data
    elif filename == 'valid':
        split = data.valid_data
    else:
        raise ValueError("Split not recognized: " + str(params.evaluate_split))

    if params.use_predicted_queries:
        filename += "_use_predicted_queries"
    else:
        filename += "_use_gold_queries"

    if full_name is None:
        full_name = os.path.join(params.logdir, filename) + params.results_note

    if params.interaction_level or params.use_predicted_queries:
        examples = data.get_all_interactions(split)
        if params.interaction_level:
            evaluate_interaction_sample(
                examples,
                model,
                name=full_name,
                metrics=FINAL_EVAL_METRICS,
                total_num=atis_data.num_utterances(split),
                database_username=params.database_username,
                database_password=params.database_password,
                database_timeout=params.database_timeout,
                use_predicted_queries=params.use_predicted_queries,
                max_generation_length=params.eval_maximum_sql_length,
                write_results=True,
                use_gpu=True,
                compute_metrics=params.compute_metrics)
        else:
            evaluate_using_predicted_queries(
                examples,
                model,
                name=full_name,
                metrics=FINAL_EVAL_METRICS,
                total_num=atis_data.num_utterances(split),
                database_username=params.database_username,
                database_password=params.database_password,
                database_timeout=params.database_timeout)
    else:
        examples = data.get_all_utterances(split)
        evaluate_utterance_sample(
            examples,
            model,
            name=full_name,
            gold_forcing=False,
            metrics=FINAL_EVAL_METRICS,
            total_num=atis_data.num_utterances(split),
            max_generation_length=params.eval_maximum_sql_length,
            database_username=params.database_username,
            database_password=params.database_password,
            database_timeout=params.database_timeout,
            write_results=True)

    database_schema = read_schema(table_schema_path)
    predictions = read_prediction(full_name + "_predictions.json")
    postprocess_db_sqls = postprocess(predictions, database_schema, True) # TODO: add token/string acc?

    postprocess_sqls = []
    for db in db_list:
        for postprocess_sql, interaction_id, turn_id in postprocess_db_sqls[db]:
            postprocess_sqls.append([postprocess_sql])

    eval_scores = eval_script_evaluate(gold_path, postprocess_sqls, db_path, "match",
                                       kmaps, bool_verbal=False, bool_predict_file=False)

    print("\nall #={} acc={:3f}, easy #={} acc={:3f}, medium #={} acc={:3f}, "
          "hard #={} acc={:3f}, extra #={} acc={:3f}".format(
        eval_scores['all']['count'], eval_scores['all']['exact'],
        eval_scores['easy']['count'], eval_scores['easy']['exact'],
        eval_scores['medium']['count'], eval_scores['medium']['exact'],
        eval_scores['hard']['count'], eval_scores['hard']['exact'],
        eval_scores['extra']['count'], eval_scores['extra']['exact']
    ))


def main():
    """Main function that trains and/or evaluates a model."""
    params = interpret_args()

    # Prepare the dataset into the proper form.
    data = atis_data.ATISDataset(params)

    model = SchemaInteractionATISModel(
        params,
        data.input_vocabulary,
        data.output_vocabulary,
        data.output_vocabulary_schema,
        data.anonymizer if params.anonymize and params.anonymization_scoring else None)

    model = model.cuda()
    model.build_optim()

    pdb.set_trace()

    last_save_file = ""

    if params.evaluate and 'valid' in params.evaluate_split:
        evaluate(model, data, params, last_save_file, split='valid')
    if params.evaluate and 'dev' in params.evaluate_split:
        evaluate(model, data, params, last_save_file, split='dev')
    if params.evaluate and 'test' in params.evaluate_split:
        evaluate(model, data, params, last_save_file, split='test')

if __name__ == "__main__":
    main()
