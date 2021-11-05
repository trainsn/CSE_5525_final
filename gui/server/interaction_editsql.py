""" The main function for interactive semantic parsing based on EditSQL. Dataset: Spider. """

import os
import sys
import numpy as np
import torch
import datetime, pytimeparse
import json
import pickle
import traceback
import subprocess
import copy
import re
from collections import defaultdict, Counter

from EditSQL.postprocess_eval import read_schema, read_prediction, postprocess, write_and_evaluate, postprocess_one
from EditSQL.eval_scripts.evaluation import evaluate_single, build_foreign_key_map_from_json, evaluate
from EditSQL.eval_scripts.evaluation import WHERE_OPS, AGG_OPS
from EditSQL.data_util import dataset_split as ds
from EditSQL.data_util.interaction import load_function
from EditSQL.data_util.utterance import Utterance

from EditSQL.logger import Logger
from EditSQL.data_util import atis_data
from EditSQL.model.schema_interaction_model import SchemaInteractionATISModel
from EditSQL.model_util import Metrics, get_progressbar, write_prediction, update_sums, construct_averages,\
    evaluate_interaction_sample, evaluate_utterance_sample, train_epoch_with_interactions, train_epoch_with_utterances
from EditSQL.world_model import WorldModel
from EditSQL.error_detector import ErrorDetectorProbability, ErrorDetectorBayesDropout, ErrorDetectorSim
from EditSQL.environment import ErrorEvaluator, UserSim, RealUser
from EditSQL.agent import Agent
from EditSQL.question_gen import QuestionGenerator
from MISP_SQL.utils import SELECT_AGG_v2, WHERE_COL, WHERE_OP, WHERE_ROOT_TERM, GROUP_COL, HAV_AGG_v2, \
    HAV_OP_v2, HAV_ROOT_TERM_v2, ORDER_AGG_v2, ORDER_DESC_ASC, ORDER_LIMIT, IUEN_v2, OUTSIDE
from user_study_utils import *

import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def real_user_interaction(example, user, agent, max_generation_length):
    with torch.no_grad():
        assert len(example.identifier.split('/')) == 2
        database_id, interaction_id = example.identifier.split('/')

        os.system('clear')  # clear screen
        print_header(bool_table_color=True)  # interface header

        print(bcolors.BOLD + "Suppose you are given some tables with the following " +
              bcolors.BLUE + "headers" + bcolors.ENDC +
              bcolors.BOLD + ":" + bcolors.ENDC)
        user.show_table(database_id)  # print table

        question = input(bcolors.BOLD + "Please type the " +
                         bcolors.PINK + "question" + bcolors.ENDC +
                         bcolors.BOLD + " you want to answer." + bcolors.ENDC + "\n").split()
        # question = example.interaction.utterances[0].original_input_seq
        input_item = agent.world_model.semparser.spider_single_turn_encoding(
            example, max_generation_length, question)

        print(bcolors.BOLD + "\nSeems you want to answer the following " +
              bcolors.PINK + "question" + bcolors.ENDC +
              bcolors.BOLD + " based on this table:" + bcolors.ENDC)
        print(bcolors.PINK + bcolors.BOLD + " ".join(question) + bcolors.ENDC + "\n")
        print(bcolors.BOLD + "To help you get the answer automatically,"
                             " the system has the following yes/no questions for you."
                             "\n(When no question prompts, please " +
              bcolors.GREEN + "continue" + bcolors.ENDC +
              bcolors.BOLD + " to the next case)\n" + bcolors.ENDC)

        start_signal = input(bcolors.BOLD + "Ready? please press '" +
                             bcolors.GREEN + "Enter" + bcolors.ENDC + bcolors.BOLD + "' to start!" + bcolors.ENDC)
        while start_signal != "":
            start_signal = input(bcolors.BOLD + "Ready? please press '" +
                                 bcolors.GREEN + "Enter" + bcolors.ENDC + bcolors.BOLD + "' to start!" + bcolors.ENDC)

        start_time = datetime.datetime.now()
        init_hyp = agent.world_model.decode(input_item, bool_verbal=False, dec_beam_size=1)[0]

        try:
            hyp, bool_exit = agent.real_user_interactive_parsing_session(
                user, input_item, init_hyp, bool_verbal=False)
        except Exception:
            print("Interaction Exception in the example!")
            hyp = init_hyp

        print("\nPredicted SQL: {}".format(" ".join(hyp.sql)))
        per_time_spent = datetime.datetime.now() - start_time
        print("Your time spent: {}".format(per_time_spent))

        # post survey
        print("-" * 50)
        print("Post-study Survey: ")
        bool_unclear = input("Is the " + bcolors.BOLD + bcolors.PINK + "initial question" +
                             bcolors.ENDC + " clear?\nPlease enter y/n: ")
        while bool_unclear not in {'y', 'n'}:
            bool_unclear = input("Is the " + bcolors.BOLD + bcolors.PINK + "initial question" +
                                 bcolors.ENDC + " clear?\nPlease enter y/n: ")

# the main function
def main(params):

    # Prepare the dataset into the proper form.
    atisdata = atis_data.ATISDataset(params)

    table_schema_path = os.path.join(params.raw_data_directory, "tables.json")
    db_path = os.path.join(os.path.dirname(params.raw_data_directory), "database/")

    # model loading
    model = SchemaInteractionATISModel(
        params,
        atisdata.input_vocabulary,
        atisdata.output_vocabulary,
        atisdata.output_vocabulary_schema,
        None)

    model.load(os.path.join(params.logdir, "model_best.pt"))
    model = model.to(device)
    model.eval()

    print("ask_structure: {}".format(params.ask_structure))
    question_generator = QuestionGenerator(bool_structure_question=params.ask_structure)

    if params.err_detector == 'any':
        error_detector = ErrorDetectorProbability(1.1)  # ask any SU
    elif params.err_detector.startswith('prob='):
        prob = float(params.err_detector[5:])
        error_detector = ErrorDetectorProbability(prob)
        print("Error Detector: probability threshold = %.3f" % prob)
        assert params.passes == 1, "Error: For prob-based evaluation, set --passes 1."
    elif params.err_detector.startswith('stddev='):
        stddev = float(params.err_detector[7:])
        error_detector = ErrorDetectorBayesDropout(stddev)
        print("Error Detector: Bayesian Dropout Stddev threshold = %.3f" % stddev)
        print("num passes: %d, dropout rate: %.3f" % (params.passes, params.dropout))
        assert params.passes > 1, "Error: For dropout-based evaluation, set --passes 10."
    elif params.err_detector == "perfect":
        error_detector = ErrorDetectorSim()
        print("Error Detector: using a simulated perfect detector.")
    else:
        raise Exception("Invalid error detector setup %s!" % params.err_detector)

    if params.num_options == 'inf':
        print("WARNING: Unlimited options!")
        num_options = np.inf
    else:
        num_options = int(params.num_options)
        print("num_options: {}".format(num_options))

    kmaps = build_foreign_key_map_from_json(table_schema_path)

    world_model = WorldModel(model, num_options, kmaps, params.passes, params.dropout,
                             bool_structure_question=params.ask_structure)

    print("friendly_agent: {}".format(params.friendly_agent))
    agent = Agent(world_model, error_detector, question_generator,
                  bool_mistake_exit=params.friendly_agent,
                  bool_structure_question=params.ask_structure)

    # environment setup: user simulator
    error_evaluator = ErrorEvaluator()

    def get_table_dict(table_data_path):
        data = json.load(open(table_data_path))
        table = dict()
        for item in data:
            table[item["db_id"]] = item
        return table

    user = RealUser(error_evaluator, get_table_dict(table_schema_path), db_path)

    # only leave job == "test_w_interaction" and user == "real"
    reorganized_data = atisdata.get_all_interactions(atisdata.valid_data)
    return reorganized_data[0], user, agent, params.eval_maximum_sql_length

