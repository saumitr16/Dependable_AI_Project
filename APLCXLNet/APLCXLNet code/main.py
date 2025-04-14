# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
source code for APLC_XLNet with FGM adversarial training and LIME explainability
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                            TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from pytorch_transformers import (WEIGHTS_NAME,
                                XLNetConfig, XLNetForMultiLabelSequenceClassification,
                                XLNetTokenizer)
import lime
from lime import lime_text
from pytorch_transformers import AdamW, WarmupLinearSchedule
from utils_multi_label import (convert_examples_to_features, output_modes, processors,
                             eval_batch, eval_precision, precision, eval_ps, psp,
                             eval_ndcg, ndcg, psndcg, metric_pk, count_parameters,
                             get_one_hot)
from multi_label_dataset import MultiLabelDataset
import shutil

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys())
                for conf in [XLNetConfig]), ())

MODEL_CLASSES = {
    'xlnet': (XLNetConfig, XLNetForMultiLabelSequenceClassification, XLNetTokenizer)
}

REWEIGHTING_TYPES = ['PW', 'PW-cb']
import torch

# Verify GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU instead")
class FGM():
    """Fast Gradient Method adversarial training"""
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.5, emb_name='word_embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
def encode_text(text, tokenizer, max_length=512):
    if not text or not isinstance(text, str):
        text = "This is a sample product description."  # fallback text
    
    # Tokenize with XLNet-specific formatting
    tokens = tokenizer.tokenize(text)
    tokens = tokens[:max_length - 2]  # Account for [CLS] and [SEP]
    
    # Add XLNet special tokens
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    
    # Convert to IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Create attention mask
    attention_mask = [1] * len(input_ids)
    
    # Padding
    padding_length = max_length - len(input_ids)
    input_ids += [tokenizer.pad_token] * padding_length
    attention_mask += [0] * padding_length
    
    return {
        "input_ids": torch.tensor([input_ids]),
        "attention_mask": torch.tensor([attention_mask])
    }
def generate_lime_explanation(model, tokenizer, device, text_sample, num_labels, num_features=10, num_samples=5):
    """Generate LIME explanation with better error handling"""
    if not text_sample or not isinstance(text_sample, str):
        text_sample = "This is a sample product description to demonstrate LIME explanations."
    
    explainer = lime.lime_text.LimeTextExplainer(
        class_names=[str(i) for i in range(num_labels)],
        split_expression=lambda x: x.split(),
        bow=False
    )
    
    def predict_proba(texts):
        model.eval()
        try:
            # Process all texts in batch
            encoded_batch = [encode_text(text, tokenizer) for text in texts]
            
            # Stack inputs
            input_ids = torch.cat([e["input_ids"] for e in encoded_batch], dim=0).to(device)
            attention_mask = torch.cat([e["attention_mask"] for e in encoded_batch], dim=0).to(device)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs[0]
                probas = torch.sigmoid(logits).cpu().numpy()
            
            # Format for LIME (2 columns per class)
            full_probas = np.zeros((len(texts), 2 * num_labels))
            for i in range(num_labels):
                full_probas[:, 2*i] = 1 - probas[:, i]  # Negative class
                full_probas[:, 2*i+1] = probas[:, i]    # Positive class
            return full_probas
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return np.zeros((len(texts), 2 * num_labels))
    
    try:
        exp = explainer.explain_instance(
            text_sample,
            predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            top_labels=min(5, num_labels)  # Ensure we don't request more labels than exist
        )
        return exp
    except Exception as e:
        print(f"Explanation failed: {str(e)}")
        return None
    

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.seed)

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    # Initialize FGM if adversarial training is enabled
    if args.adversarial:
        fgm = FGM(model)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule
    no_decay = ['bias', 'LayerNorm.weight']
    if args.model_type.lower() == 'xlnet':
        model_base = getattr(model, 'module', model)  # Handles both DataParallel and non-DataParallel

        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_base.transformer.named_parameters() if not any(nd in n for nd in no_decay)], 
             'weight_decay': args.weight_decay, "lr": args.learning_rate_x},
            {'params': [p for n, p in model_base.transformer.named_parameters() if any(nd in n for nd in no_decay)], 
             'weight_decay': 0.0, "lr": args.learning_rate_x},
            {'params': [p for n, p in model_base.sequence_summary.named_parameters() if not any(nd in n for nd in no_decay)], 
             'weight_decay': args.weight_decay, "lr": args.learning_rate_h},
            {'params': [p for n, p in model_base.sequence_summary.named_parameters() if any(nd in n for nd in no_decay)], 
             'weight_decay': 0.0, "lr": args.learning_rate_h},
            {'params': [p for n, p in model_base.AdaptiveMultiLabel.named_parameters() if not any(nd in n for nd in no_decay)], 
             'weight_decay': args.weight_decay, "lr": args.learning_rate_a},
            {'params': [p for n, p in model_base.AdaptiveMultiLabel.named_parameters() if any(nd in n for nd in no_decay)], 
             'weight_decay': 0.0, "lr": args.learning_rate_a},
        ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    cls_num_list = get_cls_num_list(train_dataset.label_ids, train_dataset.num_labels)
    c = (np.log(train_dataset.input_ids.shape[0])-1)*np.power(args.B+1, args.A)
    inv_prop = 1 + c * np.power((cls_num_list+args.B), -args.A)

    # Plot class distribution and propensity scores before training starts
    if args.local_rank in [-1, 0]:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=np.arange(len(cls_num_list)), y=cls_num_list)
        plt.title('Class Distribution')
        plt.xlabel('Class Index')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(args.output_dir, 'class_distribution.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(inv_prop, 'o-')
        plt.title('Inverse Propensity Scores')
        plt.xlabel('Class Index')
        plt.ylabel('Score')
        plt.savefig(os.path.join(args.output_dir, 'propensity_scores.png'))
        plt.close()

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    # For tracking metrics for plotting
    steps, losses, p1_scores = [], [], []

    set_seed(args)

    for num_epoch in range(int(args.num_train_epochs)):
        logger.info(" Epoch = %d", num_epoch + 1)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                            disable=args.local_rank not in [-1, 0])

        p1_list = []
        loss_list = []
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                     'attention_mask': batch[1],
                     'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                     'labels': batch[3]}

            outputs = model(**inputs)
            loss, logits = outputs[:2]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)

            # Adversarial training step
            if args.adversarial:
                fgm.attack()  # Embedding attack
                outputs_adv = model(**inputs)
                loss_adv = outputs_adv[0]  # Assuming first output is loss
                if args.n_gpu > 1:
                    loss_adv = loss_adv.mean()
                if args.gradient_accumulation_steps > 1:
                    loss_adv = loss_adv / args.gradient_accumulation_steps
                loss_adv.backward()
                fgm.restore()  # Restore embeddings

            p1 = metric_pk(logits, inputs['labels'])
            p1_list.append(p1)
            loss_list.append(loss.item())

            if step % args.logging_steps == 0:
                current_loss = np.mean(loss_list)
                current_p1 = np.mean(p1_list)
                logger.info("step {}  , loss =  {} , p1 = {} ".format(
                    step, current_loss, current_p1))
                
                # Track metrics for plotting
                if args.local_rank in [-1, 0]:
                    steps.append(global_step)
                    losses.append(current_loss)
                    p1_scores.append(current_p1)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer, inv_prop)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{:010d}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

    # Plot training metrics at the end of training
    if args.local_rank in [-1, 0] and len(steps) > 0:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(steps, losses, 'b-')
        plt.title('Training Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(steps, p1_scores, 'r-')
        plt.title('P1 Score During Training')
        plt.xlabel('Steps')
        plt.ylabel('P1 Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'training_metrics.png'))
        plt.close()

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def get_cls_num_list(labels, num_labels):
    cls_num_list = np.zeros(num_labels, dtype=float)
    for label_sample in labels:
        cls_num_list[label_sample] += 1.0
    return cls_num_list

def evaluate(args, model, tokenizer, inv_prop, prefix=""):
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(
            eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0

        num_sample = 0
        k = 5
        p_count = np.zeros((len(eval_dataloader), k))
        psp_num = np.zeros((len(eval_dataloader), k))
        psp_den = np.zeros((len(eval_dataloader), k))
        n_count = np.zeros((len(eval_dataloader), k))
        psn_num = np.zeros((len(eval_dataloader), k))
        psn_den = np.zeros((len(eval_dataloader), k))

        for i, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                         'attention_mask': batch[1],
                         'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                         'labels': batch[3]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            result = torch.sigmoid(logits)
            pred_mat = torch.topk(result, k, dim=1)[1]
            pred_mat = pred_mat.detach().cpu().numpy()
            true_mat = inputs['labels'].detach().cpu().numpy()

            num_sample += true_mat.shape[0]
            p_count[i] = precision(pred_mat, true_mat, k)
            psp_num[i], psp_den[i] = psp(pred_mat, true_mat, inv_prop, k)
            n_count[i] = ndcg(pred_mat, true_mat, k)
            psn_num[i], psn_den[i] = psndcg(pred_mat, true_mat, inv_prop, k)

        eval_loss = eval_loss / nb_eval_steps
        p1, p3, p5 = eval_precision(p_count, num_sample)
        psp1, psp3, psp5 = eval_ps(psp_num, psp_den)
        n1, n3, n5 = eval_ndcg(n_count, num_sample)
        psn1, psn3, psn5 = eval_ps(psn_num, psn_den)

        result = {'p1': p1, 'p3': p3, 'p5': p5, 'psp1': psp1, 'psp3': psp3, 'psp5': psp5,
                 'ndcg1': n1, 'ndcg3': n3, 'ndcg5': n5, 'psn1': psn1, 'psn3': psn3, 'psn5': psn5}
        results.update(result)
        if args.local_rank in [-1, 0]:
            metrics = ['p1', 'p3', 'p5', 'psp1', 'psp3', 'psp5', 
                      'ndcg1', 'ndcg3', 'ndcg5', 'psn1', 'psn3', 'psn5']
            values = [result.get(m, 0) for m in metrics]
            
            plt.figure(figsize=(15, 8))
            sns.barplot(x=metrics, y=values)
            plt.title('Evaluation Metrics')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            
            plot_path = os.path.join(eval_output_dir, f"eval_metrics_{prefix if prefix else 'final'}.png")
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved evaluation metrics plot to {plot_path}")

        print('p1 : {:.2f}   p3 : {:.2f}    p5 : {:.2f}'.format(
            p1 * 100, p3 * 100, p5 * 100))

        # Add LIME explanations if requested
        if args.explain and args.local_rank in [-1, 0]:
            try:
                # Use actual text from your dataset if possible
                sample_text = "This is a sample product description for explanation."
                
                # Get a real example if available
                if len(eval_dataset) > 0:
                    sample = eval_dataset[0]
                    sample_text = tokenizer.decode(sample[0].tolist(), skip_special_tokens=True)
                    
                exp = generate_lime_explanation(
                    model,
                    tokenizer,
                    args.device,
                    sample_text,
                    args.num_labels
                )
                
                if exp is not None:
                    explanation_file = os.path.join(args.output_dir, f"lime_explanation_{prefix if prefix else 'final'}.html")
                    exp.save_to_file(explanation_file)
                    logger.info(f"Saved LIME explanation to {explanation_file}")
                else:
                    logger.warning("LIME explanation returned None")
                    
            except Exception as e:
                logger.warning(f"Failed to generate LIME explanation: {str(e)}")

        print('p1 : {:.2f}   p3 : {:.2f}    p5 : {:.2f}'.format(p1 * 100, p3 * 100, p5 * 100))
        print('psp1 : {:.2f}   psp3 : {:.2f}    psp5 : {:.2f}'.format(psp1 * 100, psp3 * 100, psp5 * 100))
        print('ndcg1 : {:.2f}   ndcg3 : {:.2f}    ndcg5 : {:.2f}'.format(n1 * 100, n3 * 100, n5 * 100))
        print('psn1 : {:.2f}   psn3 : {:.2f}    psn5 : {:.2f}'.format(psn1 * 100, psn3 * 100, psn5 * 100))

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    
    return results

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Then add these helper functions before the main() function
def plot_training_metrics(global_steps, losses, p1_scores, output_dir):
    """Plot training loss and P1 score over time"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(global_steps, losses, 'b-')
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(global_steps, p1_scores, 'r-')
    plt.title('P1 Score During Training')
    plt.xlabel('Steps')
    plt.ylabel('P1 Score')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved training metrics plot to {plot_path}")

def plot_class_distribution(cls_num_list, output_dir):
    """Plot the class distribution"""
    plt.figure(figsize=(10, 6))
    sns.barplot(x=np.arange(len(cls_num_list)), y=cls_num_list)
    plt.title('Class Distribution')
    plt.xlabel('Class Index')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    
    plot_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved class distribution plot to {plot_path}")

def plot_propensity_scores(inv_prop, output_dir):
    """Plot the inverse propensity scores"""
    plt.figure(figsize=(10, 6))
    plt.plot(inv_prop, 'o-')
    plt.title('Inverse Propensity Scores')
    plt.xlabel('Class Index')
    plt.ylabel('Score')
    
    plot_path = os.path.join(output_dir, 'propensity_scores.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved propensity scores plot to {plot_path}")

def plot_evaluation_metrics(results, output_dir):
    """Plot the evaluation metrics"""
    metrics = ['p1', 'p3', 'p5', 'psp1', 'psp3', 'psp5', 
               'ndcg1', 'ndcg3', 'ndcg5', 'psn1', 'psn3', 'psn5']
    values = [results.get(m, 0) for m in metrics]
    
    plt.figure(figsize=(15, 8))
    sns.barplot(x=metrics, y=values)
    plt.title('Evaluation Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    plot_path = os.path.join(output_dir, 'evaluation_metrics.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved evaluation metrics plot to {plot_path}")
def get_weights(reweighting, cls_num_list, inv_prop, num_labels, beta=0.9):
    per_cls_weights = None
    if reweighting == 'PW':
        per_cls_weights = (2.0*inv_prop)-1.0
        per_cls_weights = per_cls_weights/torch.sum(per_cls_weights)
        per_cls_weights = per_cls_weights * num_labels
    elif reweighting == 'PW-cb':
        eff_num = 1.0 - torch.pow(beta, cls_num_list)
        cb_term = torch.where(cls_num_list == 0, cls_num_list, (1.0 - beta) / eff_num)
        per_cls_weights = cb_term*((2.0*inv_prop)-1.0)
        per_cls_weights = per_cls_weights/torch.sum(per_cls_weights)
        per_cls_weights = per_cls_weights * num_labels
    return per_cls_weights

def get_all_weights(args, num_data, num_labels, label_ids):
    a = args.A
    b = args.B
    beta = args.beta
    c = (np.log(num_data)-1)*np.power(b+1, a)
    cls_num_list = get_cls_num_list(label_ids, num_labels)
    cls_num_list = torch.tensor(cls_num_list, dtype=torch.float)
    cutoff_values = [0] + args.adaptive_cutoff + [cls_num_list.shape[0]]
    reweighting_factors = [None] * (len(cutoff_values)-1)
    if args.reweighting in REWEIGHTING_TYPES:
        for i in range(len(cutoff_values) - 1):
            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]
            if i == 0:
                per_cls_weights_head = torch.zeros(high_idx+len(args.adaptive_cutoff), dtype=torch.float)
                inv_prop_head = 1 + c * torch.pow((cls_num_list[low_idx:high_idx]+b), -a)
                per_cls_weights_head[low_idx:high_idx] = get_weights(
                    args.reweighting, cls_num_list[low_idx:high_idx], inv_prop_head, num_labels, beta)
            else:
                inv_prop_cluster = 1 + c * torch.pow((cls_num_list[low_idx:high_idx]+b), -a)
                per_cls_weights_cluster = get_weights(
                    args.reweighting, cls_num_list[low_idx:high_idx], inv_prop_cluster, num_labels, beta)
                reweighting_factors[i] = per_cls_weights_cluster
                inv_prop_inside_head = 1 + c * torch.pow((torch.sum(cls_num_list[low_idx:high_idx])+b), -a)
                per_cls_weights_head[cutoff_values[1]+i-1] = get_weights(args.reweighting, torch.sum(
                    cls_num_list[low_idx:high_idx]), inv_prop_inside_head, num_labels, beta)
        reweighting_factors[0] = per_cls_weights_head
    return reweighting_factors

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    processor = processors[task]()
    output_mode = output_modes[task]
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file, weights_only=False)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels(args.data_dir)
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 1,
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = [f.label_id for f in features]
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = MultiLabelDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, args)
    return dataset

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate_x", default=5e-5, type=float,
                        help="The initial learning rate for XLNet.")
    parser.add_argument("--learning_rate_h", default=1e-4, type=float,
                        help="The initial learning rate for the last hidden layer.")
    parser.add_argument("--learning_rate_a", default=1e-3, type=float,
                        help="The initial learning rate for APLC.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=1000,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', 'O3']")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='',
                        help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='',
                        help="For distant debugging.")
    parser.add_argument('--num_label', type=int, default=2,
                        help="the number of labels.")
    parser.add_argument('--pos_label', type=int, default=2,
                        help="the number of maximum labels for one sample.")
    parser.add_argument('--adaptive_cutoff', nargs='+', type=int, default=[],
                        help="the number of labels in different clusters")
    parser.add_argument('--div_value', type=float, default=2.0,
                        help="the decay factor of the dimension of the hidden state")
    parser.add_argument('--last_hidden_size', type=int, default=768,
                        help="the dimension of last hidden layer")
    parser.add_argument('--gpu', type=str, default='',
                        help="the GPUs to use ")
    parser.add_argument('--A', type=float, default=0.55,
                        help="A hyperparameter of propensity scores")
    parser.add_argument('--B', type=float, default=1.5,
                        help="B hyperparameter of propensity scores")
    parser.add_argument("--reweighting", default=None, type=str,
                        help="reweighting type")
    parser.add_argument('--beta', type=float, default=0.9,
                        help="hyper parameter of class balanced reweighting")
    parser.add_argument("--adversarial", action="store_true",
                        help="Whether to use adversarial training")
    parser.add_argument("--explain", action="store_true",
                        help="Whether to generate LIME explanations")

    args = parser.parse_args()

    assert (args.reweighting in REWEIGHTING_TYPES or args.reweighting is None), "Unknown reweighting type"
    args.output_dir = os.path.join(args.output_dir, args.reweighting if args.reweighting is not None else 'None')

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.gpu !='':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    set_seed(args)

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels(args.data_dir)
    args.num_labels = len(label_list)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                        num_labels=args.num_labels, finetuning_task=args.task_name)
    config.adaptive_cutoff = args.adaptive_cutoff
    config.div_value = args.div_value
    config.last_hidden_size = args.last_hidden_size
    config.n_token = 32000

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

    train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
    
    reweighting_factors = get_all_weights(args, train_dataset.input_ids.shape[0], train_dataset.num_labels, train_dataset.label_ids)
    
    if args.local_rank in [-1, 0] and args.do_train:
        cls_num_list = get_cls_num_list(train_dataset.label_ids, train_dataset.num_labels)
        c = (np.log(train_dataset.input_ids.shape[0])-1)*np.power(args.B+1, args.A)
        inv_prop = 1 + c * np.power((cls_num_list+args.B), -args.A)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=np.arange(len(cls_num_list)), y=cls_num_list)
        plt.title('Class Distribution')
        plt.xlabel('Class Index')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(args.output_dir, 'class_distribution.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(inv_prop, 'o-')
        plt.title('Inverse Propensity Scores')
        plt.xlabel('Class Index')
        plt.ylabel('Score')
        plt.savefig(os.path.join(args.output_dir, 'propensity_scores.png'))
        plt.close()
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), 
                                      config=config, reweighting_factors=reweighting_factors)

    print(model)
    params_1 = count_parameters(model)
    print('the number of params: ', params_1)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                        find_unused_parameters=True)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            tokenizer = tokenizer_class.from_pretrained(args.output_dir)
            train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
            reweighting_factors = get_all_weights(args, train_dataset.input_ids.shape[0], train_dataset.num_labels, train_dataset.label_ids)
            cls_num_list = get_cls_num_list(train_dataset.label_ids, train_dataset.num_labels)
            c = (np.log(train_dataset.input_ids.shape[0])-1) * np.power(args.B+1, args.A)
            inv_prop = 1 + c * np.power((cls_num_list+args.B), -args.A)

            model = model_class.from_pretrained(args.model_name_or_path, 
                                  from_tf=bool('.ckpt' in args.model_name_or_path),
                                  config=config,
                                  reweighting_factors=reweighting_factors)
            model.reweighting_factors = reweighting_factors
            model.to(device)
            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)

            result = evaluate(args, model, tokenizer, inv_prop, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

            if args.local_rank in [-1, 0] and global_step == "":
                metrics = ['p1', 'p3', 'p5', 'psp1', 'psp3', 'psp5', 
                         'ndcg1', 'ndcg3', 'ndcg5', 'psn1', 'psn3', 'psn5']
                values = [result.get(m.replace('_{}'.format(global_step), ''), 0) for m in metrics]
                
                plt.figure(figsize=(15, 8))
                sns.barplot(x=metrics, y=values)
                plt.title('Final Evaluation Metrics')
                plt.ylabel('Score')
                plt.xticks(rotation=45)
                plt.ylim(0, 1)
                plt.savefig(os.path.join(args.output_dir, 'final_metrics_summary.png'))
                plt.close()

    return results

if __name__ == "__main__":
    main()