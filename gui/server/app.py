from logging import error
from flask import Flask, jsonify,request
from flask_cors import CORS
import os
import numpy as np
import random
import argparse
import torch
import sys
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
def initial(params):

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
    global bool_structure_question, bool_mistake_exit
    bool_structure_question = params.ask_structure 
    bool_mistake_exit = params.friendly_agent
    # agent = Agent(world_model, error_detector, question_generator,
    #               bool_mistake_exit=params.friendly_agent,
    #               bool_structure_question=params.ask_structure)

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
    return reorganized_data[0], user, world_model, error_detector, question_generator, params.eval_maximum_sql_length
    


def verified_opt_selection(user, opt_question, pointer, semantic_unit, cand_semantic_units,
                            opt_answer_sheet, sel_none_of_above):
    """
    User selection.
    :param user: the user to interact with.
    :param opt_question: the question and the options to the user.
    :param pointer: the pointer to the questioned unit in the tagged sequence.
    :param semantic_unit: the questioned semantic unit.
    :param cand_semantic_units: the list of candidate semantic units.
    :param opt_answer_sheet: a dict of {user selection: meta info};
            used by user simulator to select proper choices.
    :param sel_none_of_above: the index of "none of the above".
    :return: user_selections (a list of indices indicating user selections).
    """
    print("Question: %s" % opt_question)
    user_selections = user.get_selection(pointer, opt_answer_sheet, sel_none_of_above)
    user.option_selections.append((semantic_unit[0], opt_question, user_selections))
    # save to questioned_tags
    for opt_idx, cand_su in enumerate(cand_semantic_units):
        if (opt_idx + 1) in user_selections:
            user.record_user_feedback(cand_su, 'yes', bool_qa=False)
        else:
            user.record_user_feedback(cand_su, 'no', bool_qa=False)

    return user_selections
def semantic_unit_segment(tag_seq):
    tag_item_lists, seg_pointers = [], []
    for idx, tag_item in enumerate(tag_seq):
        if tag_item[0] != OUTSIDE:
            tag_item_lists.append(tag_item)
            seg_pointers.append(idx)
    return tag_item_lists, seg_pointers
def verified_qa( user, question, answer_sheet, pointer, tag_seq):
        """
        Q&A interaction.
        :param user: the user to interact with.
        :param question: the question to the user.
        :param answer_sheet: a dict of {user response: meta info};
               used by user simulator to generate proper feedback.
        :param pointer: the pointer to the questioned unit in the tagged sequence.
        :param tag_seq: a sequence of tagged semantic units.
        :return: user_feedback.
        """
        print("Question: %s" % question)
        user_feedback = user.get_answer(pointer, answer_sheet)
        user.record_user_feedback(tag_seq[pointer], user_feedback, bool_qa=True)

        return user_feedback

def real_user_interactive_parsing_session(user, input_item, hyp, bool_verbal=False):
    """
    Interaction session, curated for real user study.
    :param user: the user to interact.
    :param input_item: the input to the semantic parser; this is specific to the base parser.
    :param hyp: the initial hypothesis generated by the non-interactive base parser.
    :param bool_verbal: set to True to print details about decoding.
    :return: hyp, True/False (whether user exits)
    """
    assert user.user_type == "real"

    def undo_execution(questioned_su, avoid_items, confirmed_items):
        assert len(tracker) >= 1, "Invalid undo!"
        hyp, start_pos = tracker.pop()

        # reset user states
        user.update_pred(hyp.tag_seq, hyp.dec_seq)

        # clear feedback after start_pos
        _tag_item_lists, _seg_pointers = semantic_unit_segment(hyp.tag_seq)
        clear_start_pointer = 0
        for clear_start_pointer in _seg_pointers:
            if clear_start_pointer >= start_pos:
                break
        clear_start_dec_idx = _tag_item_lists[_seg_pointers.index(clear_start_pointer)][-1]
        poped_keys = [k for k in avoid_items.keys() if k >= clear_start_dec_idx]
        for k in poped_keys:
            avoid_items.pop(k)
        poped_keys = [k for k in confirmed_items.keys() if k >= clear_start_dec_idx]
        for k in poped_keys:
            confirmed_items.pop(k)

        # clear the last user feedback records
        last_record = user.feedback_records[-1]
        if last_record == (questioned_su, 'undo'):
            _ = user.feedback_records.pop()
            rm_su = user.feedback_records.pop()[0]
            rm_dec_idx = rm_su[-1]
        else:
            rm_su = user.feedback_records.pop()[0]
            rm_dec_idx = rm_su[-1]
            assert rm_dec_idx == questioned_su[-1]

        rm_start_idx = len(user.feedback_records) - 1
        while rm_start_idx >= 0 and user.feedback_records[rm_start_idx][0][-1] == rm_dec_idx:
            rm_start_idx -= 1
        user.feedback_records = user.feedback_records[:rm_start_idx + 1]

        return hyp, start_pos, avoid_items, confirmed_items

    # setup
    user.update_pred(hyp.tag_seq, hyp.dec_seq)
    user.clear_counter()
    user.undo_semantic_units = []
    world_model.clear()

    # state tracker
    tracker = []    # a list of (hypothesis, starting position in tag_seq)

    # error detection
    start_pos = 0
    err_su_pointer_pairs = error_detector.detection(
        hyp.tag_seq, start_pos=start_pos, bool_return_first=True)

    while len(err_su_pointer_pairs):    # for each potential erroneous unit
        su, pointer = err_su_pointer_pairs[0]
        semantic_tag = su[0]
        print("\nSemantic Unit: {}".format(su))

        # question generation
        question, cheat_sheet = q_gen.question_generation(su, hyp.tag_seq, pointer)
        if len(question):
            # user Q&A interaction
            user_feedback = verified_qa(user, question, cheat_sheet, pointer, hyp.tag_seq)
         

            tracker.append((hyp, start_pos))

            if cheat_sheet[user_feedback][0]:   # user affirms the decision
                world_model.apply_pos_feedback(su, hyp.dec_seq, hyp.dec_seq[:su[-1]])
                start_pos = pointer + 1
            else:   # user negates the decision
                if cheat_sheet[user_feedback][1] == 0:
                    dec_seq_idx = su[-1]
                    dec_prefix = hyp.dec_seq[:dec_seq_idx]

                    # update negated items
                    dec_prefix = world_model.apply_neg_feedback(su, hyp.dec_seq, dec_prefix)

                    # perform one-step beam search to generate options
                    cand_hypotheses = world_model.decode(
                        input_item, dec_beam_size=world_model.num_options,
                        dec_prefix=dec_prefix,
                        avoid_items=world_model.avoid_items,
                        confirmed_items=world_model.confirmed_items,
                        stop_step=dec_seq_idx, bool_collect_choices=True,
                        bool_verbal=bool_verbal)

                    # prepare options
                    cand_semantic_units = []
                    for cand_hyp in cand_hypotheses:
                        cand_units, cand_pointers = semantic_unit_segment(cand_hyp.tag_seq)
                        assert cand_units[-1][0] == semantic_tag
                        cand_semantic_units.append(cand_units[-1])

                    # present options
                    opt_question, opt_answer_sheet, sel_none_of_above = q_gen.option_generation(
                        cand_semantic_units, hyp.tag_seq, pointer)

                    if user.bool_undo:
                        undo_opt = sel_none_of_above + (2 if bool_structure_question else 1)
                        opt_question = opt_question[:-1] + ";\n" + \
                                        "(%d) I want to undo my last choice!" % undo_opt

                    # user selection
                    user_selections = verified_opt_selection(
                        user, opt_question, pointer, su, cand_semantic_units, opt_answer_sheet, sel_none_of_above)

                    if user.bool_undo and user_selections == [undo_opt]:
                        user.undo_semantic_units.append((su, "Step2"))
                        hyp, start_pos, world_model.avoid_items, world_model.confirmed_items = undo_execution(
                            su, world_model.avoid_items, world_model.confirmed_items)

                        # error detection in the next turn
                        err_su_pointer_pairs = error_detector.detection(
                            hyp.tag_seq, start_pos=start_pos, bool_return_first=True)
                        continue

                    for idx in range(len(opt_answer_sheet)):    # user selection feedback incorporation
                        if idx + 1 in user_selections:
                            # update dec_prefix for components whose only choice is selected
                            dec_prefix = world_model.apply_pos_feedback(
                                cand_semantic_units[idx], cand_hypotheses[idx].dec_seq, dec_prefix)
                        else:
                            dec_prefix = world_model.apply_neg_feedback(
                                cand_semantic_units[idx], cand_hypotheses[idx].dec_seq, dec_prefix)

                    # refresh decoding
                    start_pos, hyp = world_model.refresh_decoding(
                        input_item, dec_prefix, hyp, su, pointer,
                        sel_none_of_above, user_selections,
                        bool_verbal=bool_verbal)
                    user.update_pred(hyp.tag_seq, hyp.dec_seq)

                    # a friendly agent will not ask for further feedback if any wrong decision is not resolved.
                    if bool_mistake_exit and (sel_none_of_above in user_selections or
                                                    sel_none_of_above + 1 in user_selections):
                        return hyp, False

                else:   # type 1 unit: for decisions with only yes/no choices, we "flip" the current decision
                    assert cheat_sheet[user_feedback][1] == 1
                    dec_seq_idx = su[-1]

                    dec_prefix = world_model.apply_neg_feedback(
                        su, hyp.dec_seq, hyp.dec_seq[:dec_seq_idx])
                    try:
                        hyp = world_model.decode(input_item, dec_prefix=dec_prefix,
                                                        avoid_items=world_model.avoid_items,
                                                        confirmed_items=world_model.confirmed_items,
                                                        bool_verbal=bool_verbal)[0]
                    except:
                        pass
                    user.update_pred(hyp.tag_seq, hyp.dec_seq)
                    start_pos = pointer + 1

        else:
            print("WARNING: empty question in su %s, pointer %d\n" % (su, pointer))
            start_pos = pointer + 1

        # error detection in the next turn
        err_su_pointer_pairs = error_detector.detection(
            hyp.tag_seq, start_pos=start_pos, bool_return_first=True)

        if len(err_su_pointer_pairs) == 0 and user.bool_undo:
            print("\nThe system has finished SQL synthesis. This is the predicted SQL: {}".format(hyp.sql))
            # User can undo this example
            bool_undo_example = input("Please enter if you would like to undo your selections in the previous questions (y/n)?")
            if bool_undo_example == 'y':
                hyp, start_pos, world_model.avoid_items, world_model.confirmed_items = undo_execution(
                    su, world_model.avoid_items, world_model.confirmed_items)

                # error detection in the next turn
                err_su_pointer_pairs = error_detector.detection(
                    hyp.tag_seq, start_pos=start_pos, bool_return_first=True)

    return hyp, False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

#global example,user,agent,eval_maximum_sql_length

# sanity check route
@app.route('/ping', methods=['GET'])
def ping_pong():
    return jsonify('pong!')

@app.route('/onload', methods=['GET'])
def onload():
    # print('onload function')
    global example, user, world_model, error_detector, question_generator, eval_maximum_sql_length 
    example, user, world_model, error_detector, question_generator, eval_maximum_sql_length = initial(interpret_args())
    return jsonify('yes')

@app.route('/startSession', methods=['POST'])
def startSession():
    #sentence = request.get_json()['sentence']
    #print(sentence)
    #example, user, agent, eval_maximum_sql_length = interaction_editsql.main(interpret_args())
    #real_user_interaction(example, user, agent, eval_maximum_sql_length)
    assert len(example.identifier.split('/')) == 2
    global database_id, interaction_id
    database_id, interaction_id = example.identifier.split('/')
    
    return jsonify('Please type the question you want to answer!')

@app.route('/ProcessQuestion',methods=['POST'])
def ProcessQuestion():
    global input_item
    with torch.no_grad():
        input_item = world_model.semparser.spider_single_turn_encoding(
            example, eval_maximum_sql_length, request.get_json()['question'].split())
        # print(question)
    return jsonify('To help you get the answer automatically, the system has following yes/no questions for you. Ready? Press the Enter...')


@app.route('/Enter', methods=['GET'])
def Enter():
    with torch.no_grad():
        global init_hyp, start_pos,tracker,pointer,su
        start_time = datetime.datetime.now()
        init_hyp = world_model.decode(input_item, bool_verbal=False, dec_beam_size=1)[0]
        # print('init hyp', init_hyp.tag_seq)
        # a,b = real_user_interactive_parsing_session(
        #             user, input_item, init_hyp, bool_verbal=False)
        user.update_pred(init_hyp.tag_seq, init_hyp.dec_seq)
        user.clear_counter()
        user.undo_semantic_units = []
        world_model.clear()  
        tracker = []    # a list of (hypothesis, starting position in tag_seq)

        # error detection
        
        start_pos = 0
        # print(hyp.tag_seq, start_pos)
        err_su_pointer_pairs = error_detector.detection(
            init_hyp.tag_seq, start_pos=start_pos, bool_return_first=True)

        while len(err_su_pointer_pairs):
            su, pointer = err_su_pointer_pairs[0]
            semantic_tag = su[0]
            question, cheat_sheet = question_generator.question_generation(su, init_hyp.tag_seq, pointer)
        
            return jsonify({
                'question': question,
                'cheatsheet': cheat_sheet
            })
        # try:
        #     hyp, bool_exit = real_user_interactive_parsing_session(
        #             user, input_item, init_hyp, bool_verbal=False)
        # except Exception:
        #     print("Interaction Exception in the example!")
        #     hyp = init_hyp
        # per_time_spent = datetime.datetime.now() - start_time

        # return jsonify({
        #     'sql': hyp.sql,
        #     'time': format(per_time_spent)
        # })

@app.route('/Inter1', methods=['GET'])
def Inter1():
    global start_pos, su, pointer
    with torch.no_grad():
        """
        Interaction session, curated for real user study.
        :param user: the user to interact.
        :param input_item: the input to the semantic parser; this is specific to the base parser.
        :param hyp: the initial hypothesis generated by the non-interactive base parser.
        :param bool_verbal: set to True to print details about decoding.
        :return: hyp, True/False (whether user exits)
        """
        tracker.append((init_hyp, start_pos))
        world_model.apply_pos_feedback(su, init_hyp.dec_seq, init_hyp.dec_seq[:su[-1]])
        start_pos = pointer + 1
        # print('tym1: ', len(err_su_pointer_pairs))
        err_su_pointer_pairs = error_detector.detection(
            init_hyp.tag_seq, start_pos=start_pos, bool_return_first=True)
        if len(err_su_pointer_pairs) == 0:
            return {
                'flag': 'stop',
                'sentence': 'The system has finished SQL synthesis. This is the predicted SQL',
                'sql': init_hyp.sql
            }
        else:
            su, pointer = err_su_pointer_pairs[0]
            question, cheat_sheet = q_gen.question_generation(su, hyp.tag_seq, pointer)
            return {
                'flag': 'continue',
                'sentence': question
            }
            

def interpret_args():
    """ Interprets the command line arguments, and returns a dictionary. """
    parser = argparse.ArgumentParser()

    ### Data parameters
    parser.add_argument(
        '--raw_train_filename',
        type=str,
        default='../atis_data/data/resplit/processed/train_with_tables.pkl')
    parser.add_argument(
        '--raw_dev_filename',
        type=str,
        default='../atis_data/data/resplit/processed/dev_with_tables.pkl')
    parser.add_argument(
        '--raw_validation_filename',
        type=str,
        default='../atis_data/data/resplit/processed/valid_with_tables.pkl')
    parser.add_argument(
        '--raw_test_filename',
        type=str,
        default='../atis_data/data/resplit/processed/test_with_tables.pkl')

    parser.add_argument('--data_directory', type=str, default='processed_data')

    parser.add_argument('--processed_train_filename', type=str, default='train.pkl')
    parser.add_argument('--processed_dev_filename', type=str, default='dev.pkl')
    parser.add_argument('--processed_validation_filename', type=str, default='validation.pkl')
    parser.add_argument('--processed_test_filename', type=str, default='test.pkl')

    parser.add_argument('--database_schema_filename', type=str, default=None)
    parser.add_argument('--embedding_filename', type=str, default=None)

    parser.add_argument('--input_vocabulary_filename', type=str, default='input_vocabulary.pkl')
    parser.add_argument('--output_vocabulary_filename',
                        type=str,
                        default='output_vocabulary.pkl')

    parser.add_argument('--input_key', type=str, default='nl_with_dates')

    parser.add_argument('--anonymize', type=bool, default=False)
    parser.add_argument('--anonymization_scoring', type=bool, default=False)
    parser.add_argument('--use_snippets', type=bool, default=False)

    parser.add_argument('--use_previous_query', type=bool, default=False)
    parser.add_argument('--maximum_queries', type=int, default=1)
    parser.add_argument('--use_copy_switch', type=bool, default=False)
    parser.add_argument('--use_query_attention', type=bool, default=False)

    parser.add_argument('--use_utterance_attention', type=bool, default=False)

    parser.add_argument('--freeze', type=bool, default=False)
    parser.add_argument('--scheduler', type=bool, default=False)

    parser.add_argument('--use_bert', type=bool, default=False)
    parser.add_argument("--bert_type_abb", type=str, help="Type of BERT model to load. e.g.) uS, uL, cS, cL, and mcS")
    parser.add_argument("--bert_input_version", type=str, default='v1')
    parser.add_argument('--fine_tune_bert', type=bool, default=False)
    parser.add_argument('--lr_bert', default=1e-5, type=float, help='BERT model learning rate.')

    ### Debugging/logging parameters
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--deterministic', type=bool, default=False)
    parser.add_argument('--num_train', type=int, default=-1)

    parser.add_argument('--logfile', type=str, default='log.txt')
    parser.add_argument('--results_file', type=str, default='results.txt')

    ### Model architecture
    parser.add_argument('--input_embedding_size', type=int, default=300)
    parser.add_argument('--output_embedding_size', type=int, default=300)

    parser.add_argument('--encoder_state_size', type=int, default=300)
    parser.add_argument('--decoder_state_size', type=int, default=300)

    parser.add_argument('--encoder_num_layers', type=int, default=1)
    parser.add_argument('--decoder_num_layers', type=int, default=2)
    parser.add_argument('--snippet_num_layers', type=int, default=1)

    parser.add_argument('--maximum_utterances', type=int, default=5)
    parser.add_argument('--state_positional_embeddings', type=bool, default=False)
    parser.add_argument('--positional_embedding_size', type=int, default=50)

    parser.add_argument('--snippet_age_embedding', type=bool, default=False)
    parser.add_argument('--snippet_age_embedding_size', type=int, default=64)
    parser.add_argument('--max_snippet_age_embedding', type=int, default=4)
    parser.add_argument('--previous_decoder_snippet_encoding', type=bool, default=False)

    parser.add_argument('--discourse_level_lstm', type=bool, default=False)

    parser.add_argument('--use_schema_attention', type=bool, default=False)
    parser.add_argument('--use_encoder_attention', type=bool, default=False)

    parser.add_argument('--use_schema_encoder', type=bool, default=False)
    parser.add_argument('--use_schema_self_attention', type=bool, default=False)
    parser.add_argument('--use_schema_encoder_2', type=bool, default=False)

    ### Training parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_maximum_sql_length', type=int, default=200)
    parser.add_argument('--train_evaluation_size', type=int, default=100)

    parser.add_argument('--dropout_amount', type=float, default=0.5)

    parser.add_argument('--initial_patience', type=float, default=10.)
    parser.add_argument('--patience_ratio', type=float, default=1.01)

    parser.add_argument('--initial_learning_rate', type=float, default=0.001)
    parser.add_argument('--learning_rate_ratio', type=float, default=0.8)

    parser.add_argument('--interaction_level', type=bool, default=True)
    parser.add_argument('--reweight_batch', type=bool, default=False)

    ### Setting
    # parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--train', type=int, choices=[0,1], default=0)
    parser.add_argument('--debug', type=bool, default=False)

    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--attention', type=bool, default=False)
    parser.add_argument('--enable_testing', type=bool, default=False)
    parser.add_argument('--use_predicted_queries', type=bool, default=False)
    parser.add_argument('--evaluate_split', type=str, default='dev')
    parser.add_argument('--evaluate_with_gold_forcing', type=bool, default=False)
    parser.add_argument('--eval_maximum_sql_length', type=int, default=1000)
    parser.add_argument('--results_note', type=str, default='')
    parser.add_argument('--compute_metrics', type=bool, default=False)

    parser.add_argument('--reference_results', type=str, default='')

    parser.add_argument('--interactive', type=bool, default=False)

    parser.add_argument('--database_username', type=str, default="aviarmy")
    parser.add_argument('--database_password', type=str, default="aviarmy")
    parser.add_argument('--database_timeout', type=int, default=2)

    # interaction params - Ziyu
    parser.add_argument('--job', default='test_w_interaction', choices=['test_w_interaction', 'online_learning'],
                        help='Set the job. For parser pretraining, see other scripts.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--raw_data_directory', type=str, help='The data directory of the raw spider data.')
    parser.add_argument('--num_options', type=str, default='3', help='[INTERACTION] Number of options.')
    parser.add_argument('--err_detector', type=str, default='any',
                        help='[INTERACTION] The error detector: '
                             '(1) prob=x for using policy probability threshold;'
                             '(2) stddev=x for using Bayesian dropout threshold (need to set --dropout and --passes);'
                             '(3) any for querying about every policy action;'
                             '(4) perfect for using a simulated perfect detector.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='[INTERACTION] Dropout rate for Bayesian dropout-based uncertainty analysis. '
                             'This does NOT change the dropout rate in training.')
    parser.add_argument('--passes', type=int, default=1,
                        help='[INTERACTION] Number of decoding passes for Bayesian dropout-based uncertainty analysis.')
    parser.add_argument('--friendly_agent', type=int, default=0, choices=[0, 1],
                        help='[INTERACTION] If 1, the agent will not trigger further interactions '
                             'if any wrong decision is not resolved during parsing.')
    parser.add_argument('--ask_structure', type=int, default=0, choices=[0, 1],
                        help='[INTERACTION] Set to True to allow questions about query structure '
                             '(WHERE/GROUP_COL, ORDER/HAV_AGG_v2) in NL.')

    # online learning
    parser.add_argument('--setting', type=str, default='', choices=['online_pretrain_10p', 'full_train'],
                        help='Model setting; checkpoints will be loaded accordingly.')
    parser.add_argument('--supervision', type=str, default='full_expert',
                        choices=['full_expert', 'misp_neil', 'misp_neil_perfect', 'misp_neil_pos',
                                 'bin_feedback', 'bin_feedback_expert',
                                 'self_train', 'self_train_0.5'],
                        help='[LEARNING] Online learning supervision based on different algorithms.')
    parser.add_argument('--data_seed', type=int, choices=[0, 10, 100],
                        help='[LEARNING] Seed for online learning data.')
    parser.add_argument('--start_iter', type=int, default=0, help='[LEARNING] Starting iteration in online learing.')
    parser.add_argument('--end_iter', type=int, default=-1, help='[LEARNING] Ending iteration in online learing.')
    parser.add_argument('--update_iter', type=int, default=1000,
                        help='[LEARNING] Number of iterations per parser update.')

    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    if not (args.train or args.evaluate or args.interactive or args.attention):
        raise ValueError('You need to be training or evaluating')
    if args.enable_testing and not args.evaluate:
        raise ValueError('You should evaluate the model if enabling testing')

    # Seeds for random number generation
    print("## seed: %d" % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return args

if __name__ == '__main__':
    #example, user, agent, eval_maximum_sql_length = interaction_editsql.main(interpret_args())
    #interaction_editsql.real_user_interaction(example, user, agent, eval_maximum_sql_length)
    app.run()
