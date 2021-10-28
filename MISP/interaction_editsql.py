""" The main function for interactive semantic parsing based on EditSQL. Dataset: Spider. """

import os
import sys
import numpy as np
import random
import argparse
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
from EditSQL.environment import ErrorEvaluator, UserSim, RealUser, GoldUserSim
from EditSQL.agent import Agent
from EditSQL.question_gen import QuestionGenerator
from MISP_SQL.utils import SELECT_AGG_v2, WHERE_COL, WHERE_OP, WHERE_ROOT_TERM, GROUP_COL, HAV_AGG_v2, \
    HAV_OP_v2, HAV_ROOT_TERM_v2, ORDER_AGG_v2, ORDER_DESC_ASC, ORDER_LIMIT, IUEN_v2, OUTSIDE
from user_study_utils import *

import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VALID_EVAL_METRICS = [Metrics.LOSS, Metrics.TOKEN_ACCURACY, Metrics.STRING_ACCURACY]
TRAIN_EVAL_METRICS = [Metrics.LOSS, Metrics.TOKEN_ACCURACY, Metrics.STRING_ACCURACY]
FINAL_EVAL_METRICS = [Metrics.STRING_ACCURACY, Metrics.TOKEN_ACCURACY]


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
    parser.add_argument('--user', type=str, default='sim', choices=['sim', 'gold_sim', 'real'],
                        help='[INTERACTION] User type.')
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
    parser.add_argument('--output_path', type=str, default='temp', help='[INTERACTION] Where to save outputs.')

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


def evaluation(model, data, eval_fn, valid_pred_path):
    valid_examples = data.get_all_interactions(data.valid_data)
    valid_eval_results = eval_fn(valid_examples,
                                model,
                                name=valid_pred_path,
                                metrics=FINAL_EVAL_METRICS,
                                total_num=atis_data.num_utterances(data.valid_data),
                                database_username=params.database_username,
                                database_password=params.database_password,
                                database_timeout=params.database_timeout,
                                use_predicted_queries=True,
                                max_generation_length=params.eval_maximum_sql_length,
                                write_results=True,
                                use_gpu=True,
                                compute_metrics=params.compute_metrics,
                                bool_progressbar=False)[0]
    token_accuracy = valid_eval_results[Metrics.TOKEN_ACCURACY]
    string_accuracy = valid_eval_results[Metrics.STRING_ACCURACY]

    print("## postprocess_eval...")

    database_schema = read_schema(table_schema_path)
    predictions = read_prediction(valid_pred_path + "_predictions.json")
    postprocess_db_sqls = postprocess(predictions, database_schema, True)

    postprocess_sqls = []
    for db in db_list:
        for postprocess_sql, interaction_id, turn_id in postprocess_db_sqls[db]:
            postprocess_sqls.append([postprocess_sql])

    eval_acc = evaluate(gold_path, postprocess_sqls, db_path, "match",
                        kmaps, bool_verbal=False, bool_predict_file=False)['all']['exact']
    eval_acc = float(eval_acc) * 100  # percentage

    return eval_acc, token_accuracy, string_accuracy


def extract_clause_asterisk(g_sql_toks):
    """
    This function extracts {clause keyword: tab_col_item with asterisk (*)}.
    Keywords include: SELECT/HAV/ORDER_AGG_v2.
    A tab_col_item lookds like "*" or "tab_name.*".

    The output will be used to simulate user evaluation and selections.
    The motivation is that the structured "g_sql" does not contain tab_name for *, so the simulator cannot decide the
    right decision precisely.
    :param g_sql_toks: the preprocessed gold sql tokens from EditSQL.
    :return: A dict of {clause keyword: tab_col_item with asterisk (*)}.
    """
    kw2item = defaultdict(list)

    keyword = None
    for tok in g_sql_toks:
        if tok in {'select', 'having', 'order_by', 'where', 'group_by'}:
            keyword = tok
        elif keyword in {'select', 'having', 'order_by'} and (tok == "*" or re.findall("\.\*", tok)):
            kw2item[keyword].append(tok)

    kw2item = dict(kw2item)
    for kw, item in kw2item.items():
        try:
            assert len(item) <= 1
        except:
            print("\nException in clause asterisk extraction:\ng_sql_toks: {}\nkw: {}, item: {}\n".format(
                g_sql_toks, kw, item))
        kw2item[kw] = item[0]

    return kw2item


def interaction(raw_proc_example_pairs, user, agent, max_generation_length, output_path,
                metrics=None, database_username=None, database_password=None, database_timeout=None,
                write_results=False, compute_metrics=False, bool_interaction=True):
    """ Evaluates a sample of interactions. """
    database_schema = read_schema(table_schema_path)

    def _evaluation(example, gen_sequence, raw_query):
        # check acc & add to record
        flat_pred = example.flatten_sequence(gen_sequence)
        pred_sql_str = ' '.join(flat_pred)
        assert len(example.identifier.split('/')) == 2
        database_id, interaction_id = example.identifier.split('/')
        postprocessed_sql_str = postprocess_one(pred_sql_str, database_schema[database_id])
        try:
            exact_score, partial_scores, hardness = evaluate_single(
                postprocessed_sql_str, raw_query, db_path, database_id, agent.world_model.kmaps)
        except:
            question = example.interaction.utterances[0].original_input_seq
            print("Exception in evaluate_single:\nidx: {}, db: {}, question: {}\np_str: {}\ng_str: {}\n".format(
                idx, database_id, " ".join(question), postprocessed_sql_str, raw_query))
            exact_score = 0.0
            partial_scores = "Exception"
            hardness = "Unknown"

        return exact_score, partial_scores, hardness

    if write_results:
        predictions_file = open(output_path, "w")
        print("Predicting with file: " + output_path)

    metrics_sums = {}
    for metric in metrics:
        metrics_sums[metric] = 0.

    interaction_records = []

    starting_time = datetime.datetime.now()
    count_exception, count_exit = 0, 0
    for idx, (raw_example, example) in enumerate(raw_proc_example_pairs):
        with torch.no_grad():
            pdb.set_trace()
            input_item = agent.world_model.semparser.spider_single_turn_encoding(
                example, max_generation_length)

            question = example.interaction.utterances[0].original_input_seq
            true_sql = example.interaction.utterances[0].original_gold_query
            print("\n" + "#" * 50)
            print("Example {}:".format(idx))
            print("NL input: {}".format(" ".join(question)))
            print("True SQL: {}".format(" ".join(true_sql)))

            g_sql = raw_example['sql']
            g_sql["extracted_clause_asterisk"] = extract_clause_asterisk(true_sql)
            g_sql["column_names_surface_form_to_id"] = input_item[-1].column_names_surface_form_to_id
            g_sql["base_vocab"] = agent.world_model.vocab

            try:
                hyp = agent.world_model.decode(input_item, bool_verbal=False, dec_beam_size=1)[0]
                print("-" * 50 + "\nBefore interaction: \ninitial SQL: {}".format(" ".join(hyp.sql)))
            except Exception: # tag_seq generation exception - e.g., when its syntax is wrong
                count_exception += 1
                print("Decoding Exception (count = {}) in example {}!".format(count_exception, idx))
                final_encoder_state, encoder_states, schema_states, max_generation_length, snippets, input_sequence, \
                    previous_queries, previous_query_states, input_schema = input_item
                prediction = agent.world_model.semparser.decoder(
                    final_encoder_state,
                    encoder_states,
                    schema_states,
                    max_generation_length,
                    snippets=snippets,
                    input_sequence=input_sequence,
                    previous_queries=previous_queries,
                    previous_query_states=previous_query_states,
                    input_schema=input_schema,
                    dropout_amount=0.0)
                print("-" * 50 + "\nBefore interaction: \ninitial SQL: {}".format(" ".join(prediction.sequence)))
                sequence = prediction.sequence
                probability = prediction.probability

                exact_score, partial_scores, hardness = _evaluation(example, sequence, raw_example['query'])

                record = {'nl': " ".join(question), 'true_sql': " ".join(true_sql),
                          'true_sql_i': "{}".format(g_sql),
                          'sql': "{}".format(sequence), 'dec_seq': "None",
                          'tag_seq': "None", 'logprob': "{}".format(np.log(probability)),
                          "questioned_indices": [], 'q_counter': 0,
                          'exact_score': exact_score, 'partial_scores': "{}".format(partial_scores),
                          'hardness': hardness, 'exit': False, 'exception': True, 'idx': idx}
                interaction_records.append(record)
            else:
                if not bool_interaction:
                    # print("DEBUG:")
                    # _, eval_outputs, true_sels, true_sus = user.error_evaluator.compare(
                    #     g_sql, 0, hyp.tag_seq, bool_return_true_selections=True,
                    #     bool_return_true_semantic_units=True)
                    # for unit, eval_output, true_sel, true_su in zip(hyp.tag_seq, eval_outputs, true_sels, true_sus):
                    #     print("SU: {}, EVAL: {}, TSEL: {}, TSU: {}".format(unit, eval_output, true_sel, true_su))

                    exact_score, partial_scores, hardness = _evaluation(example, hyp.sql, raw_example['query'])

                    record = {'nl': " ".join(question), 'true_sql': " ".join(true_sql),
                              'true_sql_i': "{}".format(g_sql),
                              'sql': "{}".format(hyp.sql), 'dec_seq': "{}".format(hyp.dec_seq),
                              'tag_seq': "{}".format(hyp.tag_seq), 'logprob': "{}".format(hyp.logprob),
                              "questioned_indices": [], 'q_counter': 0,
                              'exact_score': exact_score, 'partial_scores': "{}".format(partial_scores),
                              'hardness': hardness, 'exit': False, 'exception': False, 'idx': idx}
                    interaction_records.append(record)
                else:
                    try:
                        new_hyp, bool_exit = agent.interactive_parsing_session(user, input_item, g_sql, hyp,
                                                                               bool_verbal=False)
                        if bool_exit:
                            count_exit += 1
                    except Exception:
                        count_exception += 1
                        print("Interaction Exception (count = {}) in example {}!".format(count_exception, idx))
                        bool_exit = False
                    else:
                        hyp = new_hyp

                    print("-" * 50 + "\nAfter interaction: \nfinal SQL: {}".format(" ".join(hyp.sql)))

                    exact_score, partial_scores, hardness = _evaluation(example, hyp.sql, raw_example['query'])

                    record = {'nl': " ".join(question), 'true_sql': " ".join(true_sql),
                              'true_sql_i': "{}".format(g_sql),
                              'sql': hyp.sql, 'dec_seq': "{}".format(hyp.dec_seq),
                              'tag_seq': "{}".format(hyp.tag_seq), 'logprob': "{}".format(hyp.logprob),
                              'q_counter': user.q_counter,
                              'questioned_indices': user.questioned_pointers,
                              'questioned_tags': "{}".format(user.questioned_tags),
                              'feedback_records': "{}".format(user.feedback_records),
                              'exact_score': exact_score, 'partial_scores': "{}".format(partial_scores),
                              'hardness': hardness, 'exit': bool_exit, 'exception': False, 'idx': idx}
                    interaction_records.append(record)

                sequence = hyp.sql
                probability = np.exp(hyp.logprob)

        original_utt = example.interaction.utterances[0]

        gold_query = original_utt.gold_query_to_use
        original_gold_query = original_utt.original_gold_query

        gold_table = original_utt.gold_sql_results
        gold_queries = [q[0] for q in original_utt.all_gold_queries]
        gold_tables = [q[1] for q in original_utt.all_gold_queries]

        flat_sequence = example.flatten_sequence(sequence)

        if write_results:
            write_prediction(
                predictions_file,
                identifier=example.identifier,
                input_seq=question,
                probability=probability,
                prediction=sequence,
                flat_prediction=flat_sequence,
                gold_query=gold_query,
                flat_gold_queries=gold_queries,
                gold_tables=gold_tables,
                index_in_interaction=0,
                database_username=database_username,
                database_password=database_password,
                database_timeout=database_timeout,
                compute_metrics=compute_metrics)

        update_sums(metrics,
                    metrics_sums,
                    sequence,
                    flat_sequence,
                    gold_query,
                    original_gold_query,
                    database_username=database_username,
                    database_password=database_password,
                    database_timeout=database_timeout,
                    gold_table=gold_table)

        sys.stdout.flush()

    if write_results:
        predictions_file.close()
    time_spent = datetime.datetime.now() - starting_time

    # stats
    q_count = sum([item['q_counter'] for item in interaction_records])
    print("#questions: {}, #questions per example: {:.3f}.".format(
        q_count, q_count * 1.0 / len(interaction_records)))
    print("#exit: {}".format(count_exit))
    print("#time spent: {}".format(time_spent))

    eval_results = construct_averages(metrics_sums, len(raw_proc_example_pairs))
    for name, value in eval_results.items():
        print(name.name + ":\t" + "%.2f" % value)

    exact_acc = np.average([item['exact_score'] for item in interaction_records])
    print("Exact_acc: {}".format(exact_acc))

    return eval_results, interaction_records


def real_user_interaction(raw_proc_example_pairs, user, agent, max_generation_length, record_save_path):

    database_schema = read_schema(table_schema_path)

    interaction_records = []
    st = 0
    time_spent = datetime.timedelta()
    count_exception, count_exit = 0, 0

    if os.path.isfile(record_save_path):
        saved_results = json.load(open(record_save_path, 'r'))
        st = saved_results['st']
        interaction_records = saved_results['interaction_records']
        count_exit = saved_results['count_exit']
        count_exception = saved_results['count_exception']
        time_spent = datetime.timedelta(pytimeparse.parse(saved_results['time_spent']))

    for idx, (raw_example, example) in enumerate(raw_proc_example_pairs):
        if idx < st:
            continue

        with torch.no_grad():
            input_item = agent.world_model.semparser.spider_single_turn_encoding(
                example, max_generation_length)

            question = example.interaction.utterances[0].original_input_seq
            true_sql = example.interaction.utterances[0].original_gold_query

            g_sql = raw_example['sql']
            g_sql["extracted_clause_asterisk"] = extract_clause_asterisk(true_sql)
            g_sql["column_names_surface_form_to_id"] = input_item[-1].column_names_surface_form_to_id
            g_sql["base_vocab"] = agent.world_model.vocab

            assert len(example.identifier.split('/')) == 2
            database_id, interaction_id = example.identifier.split('/')

            os.system('clear')  # clear screen
            print_header(len(raw_proc_example_pairs) - idx, bool_table_color=True)  # interface header

            print(bcolors.BOLD + "Suppose you are given some tables with the following " +
                  bcolors.BLUE + "headers" + bcolors.ENDC +
                  bcolors.BOLD + ":" + bcolors.ENDC)
            user.show_table(database_id)  # print table

            print(bcolors.BOLD + "\nAnd you want to answer the following " +
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
                    user, input_item, g_sql, init_hyp, bool_verbal=False)
                bool_exception = False
                if bool_exit:
                    count_exit += 1
            except Exception:
                count_exception += 1
                print("Interaction Exception (count = {}) in example {}!".format(count_exception, idx))
                bool_exit = False
                bool_exception = True
                hyp = init_hyp

            print("\nPredicted SQL: {}".format(" ".join(hyp.sql)))
            per_time_spent = datetime.datetime.now() - start_time
            time_spent += per_time_spent
            print("Your time spent: {}".format(per_time_spent))

            # post survey
            print("-" * 50)
            print("Post-study Survey: ")
            bool_unclear = input("Is the " + bcolors.BOLD + bcolors.PINK + "initial question" +
                                     bcolors.ENDC + " clear?\nPlease enter y/n: ")
            while bool_unclear not in {'y', 'n'}:
                bool_unclear = input("Is the " + bcolors.BOLD + bcolors.PINK + "initial question" +
                                         bcolors.ENDC + " clear?\nPlease enter y/n: ")
            print("-" * 50)

            # check acc & add to record
            flat_pred = example.flatten_sequence(hyp.sql)
            pred_sql_str = ' '.join(flat_pred)
            postprocessed_sql_str = postprocess_one(pred_sql_str, database_schema[database_id])
            exact_score, partial_scores, hardness = evaluate_single(
                postprocessed_sql_str, raw_example['query'], db_path, database_id, agent.world_model.kmaps)

            g_sql.pop("base_vocab") # do not save it
            record = {'nl': " ".join(question), 'true_sql': " ".join(true_sql),
                      'true_sql_i': "{}".format(g_sql),
                      'init_sql': init_hyp.sql,
                      'sql': hyp.sql, 'dec_seq': "{}".format(hyp.dec_seq),
                      'tag_seq': "{}".format(hyp.tag_seq), 'logprob': "{}".format(hyp.logprob),
                      'exit': bool_exit, 'exception': bool_exception, 'q_counter': user.q_counter,
                      'questioned_indices': user.questioned_pointers,
                      'questioned_tags': "{}".format(user.questioned_tags),
                      'per_time_spent': str(per_time_spent), 'bool_unclear': bool_unclear,
                      'feedback_records': "{}".format(user.feedback_records),
                      'undo_semantic_units': "{}".format(user.undo_semantic_units),
                      'exact_score': exact_score, 'partial_scores': "{}".format(partial_scores), 'hardness': hardness,
                      'idx': idx}
            interaction_records.append(record)

            print("Saving records...")
            json.dump({'interaction_records': interaction_records,
                       'st': idx+1, 'time_spent': str(time_spent),
                       'count_exit': count_exit, 'count_exception': count_exception},
                      open(record_save_path, "w"), indent=4)

            end_signal = input(bcolors.GREEN + bcolors.BOLD +
                                   "Next? Press 'Enter' to continue, Ctrl+C to quit." + bcolors.ENDC)
            if end_signal != "":
                return

    print(bcolors.RED + bcolors.BOLD + "Congratulations! You have completed all your task!" + bcolors.ENDC)
    print("Your average time spent: {}".format((time_spent / len(raw_proc_example_pairs))))
    print("You exited %d times." % count_exit)
    print("%d exceptions occurred." % count_exception)


if __name__ == "__main__":
    params = interpret_args()

    # Prepare the dataset into the proper form.
    data = atis_data.ATISDataset(params)

    table_schema_path = os.path.join(params.raw_data_directory, "tables.json")
    gold_path = os.path.join(params.raw_data_directory, "dev_gold.sql")
    db_path = os.path.join(os.path.dirname(params.raw_data_directory), "database/")

    db_list = []
    with open(gold_path) as f:
        for line in f:
            line_split = line.strip().split('\t')
            if len(line_split) != 2:
                continue
            db = line.strip().split('\t')[1]
            if db not in db_list:
                db_list.append(db)

    if params.job == "online_learning" and params.supervision == 'full_train':
        model = None    # the model will be renewed immediately in online training
    else:
        # model loading
        model = SchemaInteractionATISModel(
            params,
            data.input_vocabulary,
            data.output_vocabulary,
            data.output_vocabulary_schema,
            data.anonymizer if params.anonymize and params.anonymization_scoring else None)

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

    if params.user == "real":
        def get_table_dict(table_data_path):
            data = json.load(open(table_data_path))
            table = dict()
            for item in data:
                table[item["db_id"]] = item
            return table

        user = RealUser(error_evaluator, get_table_dict(table_schema_path), db_path)
    elif params.user == "gold_sim":
        user = GoldUserSim(error_evaluator, bool_structure_question=params.ask_structure)
    else:
        user = UserSim(error_evaluator, bool_structure_question=params.ask_structure)

    # load raw data
    raw_train_examples = json.load(open(os.path.join(params.raw_data_directory, "train_reordered.json")))
    raw_valid_examples = json.load(open(os.path.join(params.raw_data_directory, "dev_reordered.json")))

    # only leave job == "test_w_interaction"
    if params.user == "real":
        user_study_indices = json.load(open(os.path.join(
            params.raw_data_directory, "user_study_indices.json"), "r"))  # random.seed(1234), 100
        reorganized_data = list(zip(raw_valid_examples, data.get_all_interactions(data.valid_data)))
        reorganized_data = [reorganized_data[idx] for idx in user_study_indices]

        real_user_interaction(reorganized_data, user, agent, params.eval_maximum_sql_length, params.output_path)

    else:
        reorganized_data = list(zip(raw_valid_examples, data.get_all_interactions(data.valid_data)))

        eval_file = params.output_path[:-5] + "_prediction.json"
        eval_results, interaction_records = interaction(
            reorganized_data, user, agent,
            params.eval_maximum_sql_length,
            eval_file,
            metrics=FINAL_EVAL_METRICS,
            database_username=params.database_username,
            database_password=params.database_password,
            database_timeout=params.database_timeout,
            write_results=True,
            compute_metrics=params.compute_metrics,
            bool_interaction=True)
        json.dump(interaction_records, open(params.output_path, "w"), indent=4)
