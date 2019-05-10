import csv
import os
import logging
import argparse
import random
import constants
import datetime
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm, trange
from typing import List
from collections import defaultdict

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from data_processors import (
    TOEFL11Processor,
    RedditInDomainDataProcessor,
    RedditOutOfDomainDataProcessor,
    AllOfRedditDataProcessor,
)

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from sklearn.metrics import f1_score, precision_score, recall_score

from plot_confusion_matrix import plot_confusion_matrix

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, guid, input_ids, input_mask, segment_ids, label_id):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    num_unkown = 0
    unk_id = tokenizer.vocab['[UNK]']
    logger.info(f'UNK id is: {unk_id}')

    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None

        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        num_unkown += input_ids.count(unk_id)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(guid=example.guid,
                              input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features, num_unkown


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def get_timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d_%H:%M')

def get_eval_folder_name(args):
    return f'seq_{args.max_seq_length}_batch_{args.train_batch_size }_epochs_{args.num_train_epochs}_lr_{args.learning_rate}'

def save_csv(all_guids, all_inputs, all_outputs, all_predicted_logits, label_list, output_eval_file):
    prediction_dict = {
        "guid" : all_guids,
        "input" : all_inputs,
        "output" : all_outputs,
        "input_label" : [label_list[int(i)] for i in all_inputs],
        "output_label" : [label_list[int(i)] for i in all_outputs],
    }
    for i in range(len(label_list)):
        prediction_dict[label_list[i]] = all_predicted_logits[i]

    pd.DataFrame(prediction_dict).to_csv(f'{output_eval_file}.csv', index=False)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default="./results/",
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--vocab_file',
                        type=str, default=None,
                        help="Path to the vocab file to use for the tokenizer.")

    parser.add_argument('--cross_validation_fold',
                        type=int, default=None,
                        help="If cross validating, this tells the model what fold to use.")

    parser.add_argument('--binary_target_lang',
                        type=str, default=None,
                        help="If training a binary classifier for one language, this is the target label")

    args = parser.parse_args()

    processors = {
        "toefl11": TOEFL11Processor,
        "redditl2": RedditInDomainDataProcessor,
        "out-of-domain-redditl2": RedditOutOfDomainDataProcessor,
        "all-of-reddit": AllOfRedditDataProcessor,
    }

    num_labels_task = {
        "toefl11": 11,
        "redditl2": 23,
        "out-of-domain-redditl2": 23,
        "all-of-reddit": 23,
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    if task_name.lower() == 'redditl2' and not args.cross_validation_fold:
        raise ValueError('The reddit data set cannot be used without cross validation')

    if args.cross_validation_fold:
        logger.info('Using cross-validation')
        processor = processors[task_name](args.cross_validation_fold)
    elif args.binary_target_lang:
        logger.info(f'Target binary language is: {args.binary_target_lang}')
        processor = processors[task_name](args.binary_target_lang)
    else:
        processor = processors[task_name]()

    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    if args.vocab_file:
        tokenizer = BertTokenizer.from_pretrained(args.vocab_file, do_lower_case=args.do_lower_case)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
              cache_dir=cache_dir,
              num_labels = num_labels)

    if args.fp16:
        model.half()

    model.to(device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)

    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            warmup=args.warmup_proportion,
                            t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if args.do_train:
        train_features, num_unkown = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)

        logger.info(f'Number of "[UNK]" in training data: {num_unkown}')

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        if args.cross_validation_fold:
            logger.info(f'Cross validation fold: {args.cross_validation_fold}')

        all_guids = [f.guid for f in train_features]
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

    timestamp = get_timestamp()
    model_foldername = f'{timestamp}_seq_{args.max_seq_length}_batch_{args.train_batch_size }_epochs_{args.num_train_epochs}_lr_{args.learning_rate}'
    if args.binary_target_lang:
        model_foldername = args.binary_target_lang + '_' + model_foldername

    full_path = f'{args.output_dir}/{model_foldername}'

    if not args.cross_validation_fold:
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)

        os.mkdir(full_path)

    # Save training data outputs for meta-classifiers
    logger.info('Saving final training outputs')

    num_train_examples = len(train_data)
    all_inputs = np.zeros((num_train_examples))
    all_outputs = np.zeros((num_train_examples))
    all_predicted_logits = np.zeros((num_train_examples, len(label_list)))
    data_idx = 0

    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.detach().cpu().numpy()

        outputs = np.argmax(logits, axis=1)

        all_inputs[data_idx:data_idx + args.train_batch_size] = label_ids
        all_outputs[data_idx:data_idx + args.train_batch_size] = outputs
        all_predicted_logits[data_idx:data_idx + args.train_batch_size] = logits
        data_idx += args.train_batch_size

    csv_filename = full_path + '_train_' + f'fold_{args.cross_validation_fold}'
    save_csv(all_guids, all_inputs, all_outputs, all_predicted_logits.T, label_list, csv_filename)


    # Save a trained model
    if args.do_train:
        if args.cross_validation_fold:
            logger.info('Cross validation is used, so no model is saved.')
        else:
            # Save a trained model and the associated configuration
            logger.info('Saving model!')
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(full_path, WEIGHTS_NAME)
            torch.save(model_to_save.state_dict(), output_model_file)
            output_config_file = os.path.join(full_path, CONFIG_NAME)
            with open(output_config_file, 'w') as f:
                f.write(model_to_save.config.to_json_string())

            # Load a trained model and config that you have fine-tuned
            config = BertConfig(output_config_file)
            model = BertForSequenceClassification(config, num_labels=num_labels)
            model.load_state_dict(torch.load(output_model_file))
    else:
        model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)

    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)

        # Keep track of all train ids, to make sure non are used for evaluation
        train_ids = set([input_example.guid for input_example in train_examples])

        eval_ids = [input_example.guid for input_example in eval_examples]
        for eval_id in eval_ids:
            if eval_id in train_ids:
                raise ValueError(f'{eval_id} is present both in test and training set.')

        # Clean memory
        del eval_ids
        del train_ids
        logger.info('Good news: No training guids found in test guids.')

        eval_features, num_unkown = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)

        logger.info(f'Number of "[UNK]" in test data: {num_unkown}')
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_guids = [f.guid for f in eval_features]
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        num_eval_examples = len(eval_data)
        all_inputs = np.zeros((num_eval_examples))
        all_outputs = np.zeros((num_eval_examples))
        all_predicted_logits = np.zeros((num_eval_examples, len(label_list)))
        data_idx = 0
 
        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()

            outputs = np.argmax(logits, axis=1)

            all_inputs[data_idx:data_idx + args.eval_batch_size] = label_ids
            all_outputs[data_idx:data_idx + args.eval_batch_size] = outputs
            all_predicted_logits[data_idx:data_idx + args.eval_batch_size] = logits
            data_idx += args.eval_batch_size

            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        f1 = f1_score(all_inputs, all_outputs, average='macro')
        precision = precision_score(all_inputs, all_outputs, average='macro')
        recall = recall_score(all_inputs, all_outputs, average='macro')
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        loss = tr_loss/nb_tr_steps if args.do_train else None
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'f1': f1,
                  'precision': precision,
                  'recall': recall,
                  'global_step': global_step,
                  'loss': loss}


        if args.cross_validation_fold:
            eval_foldername = get_eval_folder_name(args)
            folder_path = os.path.join(args.output_dir, eval_foldername)
            os.makedirs(folder_path, exist_ok=True)
            output_eval_file = folder_path + f'/fold_{args.cross_validation_fold}'

        else:
            timestamp = get_timestamp()
            eval_filename = f'{timestamp}_acc{eval_accuracy:.3f}_seq_{args.max_seq_length}_batch_{args.train_batch_size }_epochs_{args.num_train_epochs}_lr_{args.learning_rate}'
            if args.binary_target_lang:
                eval_filename = args.binary_target_lang + '_' + eval_filename

            output_eval_file = os.path.join(args.output_dir, f'{eval_filename}')

            # plot_confusion_matrix(all_inputs, all_outputs, classes=label_list, normalize=True,
            #                     title='Normalized confusion matrix')
            # np.set_printoptions(precision=2)
            # plt.savefig(f'./out/confusion_matrices/{args.bert_model}_{eval_filename}.png')


        with open(output_eval_file + '.txt', "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        save_csv(all_guids, all_inputs, all_outputs, all_predicted_logits.T, label_list, output_eval_file)

if __name__ == "__main__":
    main()
