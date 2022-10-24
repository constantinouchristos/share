import pandas as pd
import numpy as np 
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, TensorDataset
from tqdm import trange, tqdm
import json
from transformers import (BertConfig,
                          BertTokenizer,
                          BertForSequenceClassification,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          TrainingArguments)
from sklearn.metrics import f1_score,balanced_accuracy_score
import shutil
from IPython.display import display as dis
from support import add_method,get_attrs,get_methods,help


@add_method('help')
@add_method('get_attrs')
@add_method('get_methods')
class TrainerState:
    
    def __init__(self,args,t_total):
        
        self.args = args
        self.t_total = t_total
        
        self.best_model_chepoint_step = None
        self.global_step = None
        self.best_eval_score = -99999
        self.num_total_evaluations = 30
        self.continue_training = True
        self.best_model_chepoint = None
        
        self.set_evaluation_strategy()
        
    def get_val(self,name):
        
        return getattr(self,name)
    
    def update_val(self,name,value):
        
        setattr(self,name,value)
        
        
    def set_evaluation_strategy(self):
        
        steps_to_delay = self.t_total / self.args.num_train_epochs
        
        
        
        self.args.eval_delay = steps_to_delay
        
        remaining_steps_for_eval = self.t_total - steps_to_delay
        
        self.args.eval_steps = remaining_steps_for_eval // self.num_total_evaluations
        
        self.early_stopper = EarlyStoppingCallback(self.args)
        
    def update_status(self):
        
        if self.best_eval_score != self.early_stopper.current_best:
            
            self.best_eval_score = self.early_stopper.current_best
            self.best_model_chepoint_step = self.early_stopper.best_step
            
        
        if self.early_stopper.stop_training:
            
            self.continue_training = False

class EarlyStoppingCallback:
    
    
    def __init__(self,
                 args=None,
                 early_stopping_patience=10,
                 early_stopping_threshold=0.0
                ):
        self.args = args
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_patience_counter = 0
        
        self.score_to_optimize = self.args.metric_for_best_model
        self.history = {}
        self.current_best = None
        self.stop_training = False
    
    def update_state(self,
                     step=None,
                     metrics=None):
        
        
        if self.current_best is None:
            self.current_best = metrics[self.score_to_optimize]
            self.best_step = step
            for key_metr in  metrics:
                
                    self.history[key_metr] = [metrics[key_metr]] 
            self.history['step'] = [step]
            
        else:
            
            for key_metr in  metrics:
                    self.history[key_metr].append(metrics[key_metr])

            self.history['step'].append(step)
            
            if metrics[self.score_to_optimize] > self.current_best:
                
                self.current_best = metrics[self.score_to_optimize]
                self.best_step = step
                self.early_stopping_patience_counter = 0
            else:
                self.early_stopping_patience_counter += 1
                
            if self.early_stopping_patience_counter > self.early_stopping_patience:
                self.stop_training = True
                
        self.generate_history_table()
                
    def generate_history_table(self):
        
        
        self.df_history = pd.DataFrame(self.history)
                
    
@add_method('help')
@add_method('get_attrs')
@add_method('get_methods')
class Trainer:
    
    
    def __init__(self,
                 train_dataset=None,
                 eval_dataset=None,
                 test_dataset=None,
                 model=None,
                 args=None,
                 device='cpu',
                 swag_model=None
                 
                ):
        
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.args = args
        self.device = device
        
        # swag model
        if swag_model is not None:
            self.swag_model = swag_model
            self.swag_mode = True
            self.n_collect = 0
            self.n_collection_steps = 0
            self.freq_step_collect = 5
        else:
            self.swag_mode = False
        
        
    def get_critirion(self):
        
        return nn.CrossEntropyLoss()
        
        
        
    def get_dataloader(self,
                       dataset,
                       dat_type='train'
                      ):
        
        bs = self.args.per_device_train_batch_size,
        
        if dat_type == 'eval':

            sampler = SequentialSampler(dataset)
            bs = self.args.per_device_eval_batch_size
        else:
            sampler = RandomSampler(dataset)
            bs = self.args.per_device_train_batch_size
            
        dl = DataLoader(dataset,
                        batch_size=bs,
                        sampler=sampler)
        
        return dl
    
    def get_optimiser(self,
                      model,
                      args
                     ):
        
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, 
                          lr=args.learning_rate, 
                          eps=args.adam_epsilon)
        
        return optimizer
    
    
    def get_scheduler(self,
                      optimizer,
                      t_total,
                      args
                     ):
        
        
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)
        
        return scheduler
        
        
    def train(self):
        
        training_args = self.args
        
        # prepare train dl
        train_dl = self.get_dataloader(self.train_dataset,
                                       dat_type='train'
                                      )
        
        # total optimisation steps
        t_total = len(train_dl) // training_args.gradient_accumulation_steps * training_args.num_train_epochs
        
        if self.swag_mode:
            
            self.step_start_swag = t_total / training_args.num_train_epochs
        
        
        optimizer = self.get_optimiser(self.model,
                                       training_args)
        
        
        scheduler = self.get_scheduler(optimizer,
                                       t_total,
                                       training_args
                                      )
        
        
        critirion = self.get_critirion()
    
        logging_steps = 20
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        
        self.trainer_state = TrainerState(training_args,t_total)
        self.trainer_state.update_val('global_step',global_step)
        
        self.model.to(self.device)
        self.model.zero_grad()

        train_iterator = trange(int(training_args.num_train_epochs), desc="Epoch")
        
        
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dl, desc="Iteration")

            for step, batch in enumerate(epoch_iterator):
                self.model.train()

                inputs = {"input_ids":batch[0],
                                     "attention_mask":batch[1],
                                     "token_type_ids":batch[2],
                                    }

                labels = batch[3].to(self.device)

                inputs = {k: v.to(self.device) if v is not None else None for k, v in inputs.items()}
                outputs = self.model(**inputs)

                logits = outputs.logits


                loss = critirion(logits,labels)

                if training_args.gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % training_args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), training_args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if logging_steps > 0 and global_step % logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs['learning_rate'] = learning_rate_scalar
                        logs['loss'] = loss_scalar
                        logging_loss = tr_loss

                        print(json.dumps({**logs, **{'step': global_step}}))
                        
                    if (self.trainer_state.args.eval_steps > 0 
                        and  global_step > self.trainer_state.args.eval_delay  
                        and global_step % self.trainer_state.args.eval_steps == 0
                       ):
                        
                        print('Evaluating ...')
                        
                        metrics = self.eval(self.eval_dataset)
                        
                        self.trainer_state.early_stopper.update_state(global_step,
                                                                      metrics)
                        
                        if self.trainer_state.best_eval_score != self.trainer_state.early_stopper.current_best:
                            
                            self.save_delete_model(checkpoint=self.trainer_state.early_stopper.best_step)
                        
                        self.trainer_state.update_status()
                        
                        print('evaluation results:')
                        dis(self.trainer_state.early_stopper.df_history)
                        
                    if self.swag_mode and global_step > self.step_start_swag  :
                        
                        self.n_collection_steps += 1
                        
                        if self.n_collection_steps % self.freq_step_collect == 0:
                            print(f'collecting swag model at step: {global_step}')
                            self.swag_model.collect_model(self.model.cpu())
                            self.model.to(self.device)
                            

                    
                if not self.trainer_state.continue_training:
                    break
                    
            if not self.trainer_state.continue_training:
                break
        
        if training_args.load_best_model_at_end:
            self.load_model()
            
    def calculate_metrics(self,all_true,all_preds):

        return {"f1": f1_score(all_true,all_preds,average='weighted'),
                "b_acc": balanced_accuracy_score(all_true,all_preds)
               }

    
    def eval(self,
             dataset):
        
        
        data_loader = self.get_dataloader(dataset,
                                          dat_type='eval'
                                         )


        model = self.model
        model.to(self.device)

        model.eval()

        all_true = []
        all_preds = []

        for step, batch in enumerate(data_loader):

            inputs = {"input_ids":batch[0],
                      "attention_mask":batch[1],
                      "token_type_ids":batch[2],
                     }

            labels = batch[3]

            inputs = {k: v.to(self.device) if v is not None else None for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits

            all_true.extend(labels.tolist())
            all_preds.extend(logits.cpu().numpy().argmax(axis=1).tolist())

        all_true = np.hstack(all_true)
        all_preds = np.hstack(all_preds)
        
        
        metrics = self.calculate_metrics(all_true,all_preds)

        return metrics
    
    
    def save_delete_model(self,checkpoint=None):
        
        checkpoints_path_general = os.path.join(self.args.output_dir,'checkpoints')
        
        if not os.path.exists(checkpoints_path_general):
            os.makedirs(checkpoints_path_general)
        
        # remove previous checkpoints
        all_past_check_points = [
            os.path.join(checkpoints_path_general,i) 
            for i in os.listdir(checkpoints_path_general)
        ]
        
        for path_previous_check in all_past_check_points:
            shutil.rmtree(path_previous_check)
        
        
        PATH = os.path.join(checkpoints_path_general,f'checkpoint_{checkpoint}')
        
        
        if not os.path.exists(PATH):
            os.makedirs(PATH)
            
        path_to_save = os.path.join(PATH,'pytoch_model.bin')
        
        torch.save(self.model.state_dict(), path_to_save)
        

        self.trainer_state.best_model_chepoint = path_to_save

    
    def load_model(self,checkpoint=None):
        
        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint))
        elif self.trainer_state.best_model_chepoint is not None:
            self.model.load_state_dict(torch.load(self.trainer_state.best_model_chepoint))
        else:
            pass
    
        
