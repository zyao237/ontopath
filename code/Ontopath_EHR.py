import numpy as np
import sys
import random
import pandas as pd
import math
from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
from tqdm import trange, tqdm
import pickle
import copy
import argparse
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score
from OntopathDataset_EHR import OntopathDataset, padding_collate_fn
from Transformer_EHR import Transformer
import logging
import Utils as ut
from ContrastiveLoss import ConLoss

class Ontopath(nn.Module):
    def __init__(self, patient_num, drug_num, atc_num, icd_num, patient_demo_dim, drug_se_dim, atc_path_length=5, icd_path_length=4,
                 dim_emb=50, dropout=0.0, dim_network=50, dim_output=1, batch_first=True, bidirectional=False,
                 initializer='normal', predictor='dot', transformer_order=True):
        super(Ontopath, self).__init__()
        self.batch_first = batch_first
        self.icd_path_length = icd_path_length

        self.icd_embedding = nn.Embedding(icd_num, dim_emb)
        self.atc_embedding = nn.Embedding(atc_num, dim_emb)

        self.transformer = Transformer(d_model=dim_emb, N=1, heads=2, dropout=dropout, order=transformer_order)

        if bidirectional:
            dim_network = int(dim_network/2)
        self.rnn_patient = nn.GRU(input_size=dim_emb, hidden_size=dim_network, num_layers=1, batch_first=self.batch_first, bidirectional=bidirectional)
        self.rnn_drug = nn.GRU(input_size=dim_emb, hidden_size=dim_network, num_layers=1, batch_first=self.batch_first, bidirectional=bidirectional)

        if bidirectional:
            dim_network = int(dim_network*2)
        self.C = nn.Parameter(torch.empty((dim_network, dim_network), requires_grad=True))
        self.register_parameter(name='attention', param=self.C)

        self.att_activate = nn.Tanh()
        self.predict_activate = nn.ReLU()

        self.predictor = predictor
        if predictor == 'mlp':
            self.predict_layer = nn.Sequential(
                nn.Linear(in_features=dim_emb * 2 + dim_network * 2, out_features=dim_emb * 2),
                self.predict_activate,
                nn.Linear(in_features=dim_emb * 2, out_features=dim_emb),
                self.predict_activate,
                nn.Linear(in_features=dim_emb, out_features=dim_output)
            )
        else:
            self.predict_layer = nn.Sequential(
                nn.Linear(in_features=dim_emb * 2, out_features=dim_output)
            )

        self.patient_demo_encoder = nn.Sequential(
            nn.Linear(in_features=patient_demo_dim, out_features=dim_emb)
        )

        self.drug_se_encoder = nn.Sequential(
            nn.Linear(in_features=drug_se_dim, out_features=dim_emb)
        )

        self._init_weight(initializer)

    def _gmf_prediction(self, health_rep_batch, path_rep_batch, se_batch):
        health_se = health_rep_batch * se_batch
        health_drug = health_rep_batch * path_rep_batch
        return torch.cat([health_drug, health_se], dim=1)

    def _init_weight(self, initializer):
        assert initializer in ['normal', 'uniform']
        if initializer=='normal':
            init.normal_(self.atc_embedding.weight, std=0.01)
            init.normal_(self.icd_embedding.weight, std=0.01)
            init.normal_(self.C, std=0.01)
            for p in self.transformer.parameters():
                if p.dim() > 1:
                    init.normal_(p, std=0.01)
                else:
                    init.constant_(p, 0)
        elif initializer=='uniform':
            init.normal_(self.atc_embedding.weight, std=0.01)
            init.normal_(self.icd_embedding.weight, std=0.01)
            init.xavier_uniform_(self.C)
            for p in self.transformer.parameters():
                if p.dim() > 1:
                    init.xavier_uniform_(p)
                else:
                    init.constant_(p, 0)
        self._init_nn_seq(self.predict_layer, initializer)
        self._init_nn_seq(self.patient_demo_encoder, initializer)
        self._init_nn_seq(self.drug_se_encoder, initializer)

    def _init_nn_seq(self, seq, initializer):
        for m in seq:
            if isinstance(m, nn.Linear):
                if initializer=='normal':
                    init.normal_(m.weight, std=0.01)
                    init.constant_(m.bias, 0)
                elif initializer=='uniform':
                    init.xavier_uniform_(m.weight)
                    init.constant_(m.bias, 0)

    def forward(self, patient_batch, drug_batch, demo_batch, se_batch, path_batch, ehr_batch, ehr_mask):
        patient_emb = self.patient_demo_encoder(demo_batch)
        drug_emb = self.drug_se_encoder(se_batch)

        ehr_emb = self.icd_embedding(ehr_batch)
        patient_emb_repeat = torch.unsqueeze(torch.repeat_interleave(patient_emb, self.icd_path_length, dim=0), dim=1)
        patient_health_emb = self.transformer(ehr_emb, patient_emb_repeat, ehr_mask)
        drug_path_emb = self.atc_embedding(path_batch)

        g_health, _ = self.rnn_patient(patient_health_emb.view(patient_emb.size(0), -1, patient_emb.size(1)))
        g_path, _ = self.rnn_drug(drug_path_emb)

        att_mat_raw = torch.matmul(torch.matmul(g_health, self.C), torch.transpose(g_path, 1, 2))
        att_mat = self.att_activate(att_mat_raw)
        health_att_logit = torch.max(att_mat, 2, keepdim=True, out=None)[0]
        path_att_logit = torch.max(att_mat, 1, keepdim=True, out=None)[0]
        path_att_logit = torch.transpose(path_att_logit, 1, 2) # make it back to (batch, seq_len, d)
        health_att = self._masked_softmax(health_att_logit, 1)
        path_att = self._masked_softmax(path_att_logit, 1)

        health_rep = torch.sum(g_health * health_att, dim=1, keepdim=False)
        path_rep = torch.sum(g_path * path_att, dim=1, keepdim=False)

        if self.predictor == 'mlp':
            # MLP Neural CF output
            health_pat_rep = torch.cat([health_rep, patient_emb], dim=1)
            path_drug_rep = torch.cat([path_rep, drug_emb], dim=1)
            concat_vector = torch.cat([health_pat_rep, path_drug_rep], dim=1)
            output_batch = self.predict_layer(concat_vector)
        else:
            # dot product CF output
            gmf_output = self._gmf_prediction(health_rep, path_rep, drug_emb)
            output_batch = self.predict_layer(gmf_output)

        prediction_batch = torch.sigmoid(output_batch)
        # return prediction_batch
        return health_rep, path_rep, prediction_batch

    def _masked_softmax(self, batch_tensor, dim, mask=None):
        exp = torch.exp(batch_tensor)
        if mask is not None:
            exp = exp * mask
        sum_masked_exp = torch.sum(exp, dim=dim, keepdim=True)
        return exp / sum_masked_exp

class Pipeline:
    def __init__(self, params, data):
        self.params = params
        logging.info(self.params)

        # config
        np.random.seed(self.params['seed'])
        torch.manual_seed(self.params['seed'])

        trainset = OntopathDataset(data['trainset'], data['drug_path'], data['icd_path'], data['patient_demo'], data['drug_se'],
                                  data['index_data'], train=True)
        testset = OntopathDataset(data['testset'], data['drug_path'], data['icd_path'], data['patient_demo'], data['drug_se'],
                                  data['index_data'], train=False)

        self.train_loader = DataLoader(dataset=trainset, batch_size=self.params['batch_size'], shuffle=True, collate_fn=padding_collate_fn, num_workers=4, drop_last=True)
        self.test_loader = DataLoader(dataset=testset, batch_size=self.params['batch_size'], shuffle=False, collate_fn=padding_collate_fn, drop_last=True)

        # print('===> data loading complete')
        logging.info("patient: {}".format(self.params['n_patient']))
        logging.info("drug: {}".format(self.params['n_drug']))
        logging.info("atc: {}".format(self.params['n_atc']))
        logging.info("icd: {}".format(self.params['n_icd']))
        logging.info("data for training/testing: {}/{}".format(len(trainset)*(1+self.params['n_negative_sample']), len(testset)))

        logging.info('===> initialization complete')

        self.finetuning_criterion = nn.BCELoss()
        device = self._get_device()
        self.pretrain_criterion = ConLoss(device, batch_size=self.params['batch_size'], temperature=self.params['softmax_temperature'], use_cosine_similarity=False)

        os.environ["CUDA_VISIBLE_DEVICES"] = self.params['gpu']
        cudnn.benchmark = True

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    # load or initialize model. Initialize a new one for training, load one for testing
    def _model_load(self, model_wts=None):
        self.model = Ontopath(
            patient_num=self.params['n_patient'],
            drug_num=self.params['n_drug'],
            atc_num=self.params['n_atc'],
            icd_num=self.params['n_icd'],
            patient_demo_dim=self.params['demo_dim'],
            drug_se_dim=self.params['se_dim'],
            dim_emb=self.params['K'],
            dim_network=self.params['K'],
            dropout=self.params['dropout'],
            batch_first=True,
            bidirectional=self.params['bidirectional'],
            initializer=self.params['initializer'],
            predictor=self.params['predictor'],
            transformer_order=self.params['transformer_order']
        )
        logging.info('===> Model built, initialized using {}'.format(self.params['initializer']))
        if model_wts is not None:
            self.model.load_state_dict(model_wts)
            print('===> Load a model!')

    def train(self, model_params=None, pretrain=False):
        # Create model
        if model_params != None:
            self._model_load(model_params)
        else:
            self._model_load()
        if self.params['gpu'] is not "":
            self.model.cuda()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'], weight_decay=self.params['L2_norm'])  # , betas=(0.1, 0.001), eps=1e-8
        # optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.params['lr'], weight_decay=self.params['L2_norm'])
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params['lr'], weight_decay=self.params['L2_norm'], momentum=0.9)

        best_model_wts = copy.deepcopy(self.model.state_dict())
        train_losses = []

        best_loss = sys.float_info.max

        # best_results = ''
        if pretrain:
            epoch_num = self.params['pretrain_epochs']
            logging.info("Training model - pretrain with total {} epochs".format(epoch_num))
            cur_criterion = self.pretrain_criterion
        else:
            epoch_num = self.params['epochs']
            logging.info("Training model - finetuning with total {} epochs".format(epoch_num))
            cur_criterion = self.finetuning_criterion

        for ei in trange(epoch_num, desc="Epochs"):
            result_dict = self.epoch(self.train_loader, criterion=cur_criterion, optimizer=optimizer, train=True, pretrain=pretrain)
            train_y_true, train_y_pred, train_loss = result_dict['label'], result_dict['prediction'], result_dict['loss']
            if self.params['gpu'] is not "":
                train_y_true = train_y_true.cpu()
                train_y_pred = train_y_pred.cpu()
            # regression
            # train_r2 = r2_score(train_y_true.detach().numpy(), train_y_pred.detach().numpy())
            # train_mse = mean_squared_error(train_y_true.detach().numpy(), train_y_pred.detach().numpy())

            # classification
            # train_accuracy = accuracy_score(train_y_true.numpy(), np.round(train_y_pred.numpy()))
            # train_precision = precision_score(train_y_true.numpy(), np.round(train_y_pred.numpy()))

            train_losses.append(train_loss)

            if (ei+1) % 1 == 0:
                test_metrics = self.test(self.test_loader, cur_criterion, pretrain, topk=[1,2,3,4,5], atc_level=0)
                test_results = "- Loss test: {1:.4f}" \
                               "\nRank - HR@1: {2:.4f}, HR@2: {3:.4f}, HR@3: {4:.4f}, HR@4: {5:.4f}, HR@5: {6:.4f}" \
                               "\nRank - NDCG@1: {7:.4f}, NDCG@2: {8:.4f}, NDCG@3: {9:.4f}, NDCG@4: {10:.4f}, NDCG@5: {11:.4f}" \
                    .format(ei, test_metrics['test_loss'],
                            test_metrics['rank']['topk_hit'][1], test_metrics['rank']['topk_hit'][2], test_metrics['rank']['topk_hit'][3],
                            test_metrics['rank']['topk_hit'][4], test_metrics['rank']['topk_hit'][5],
                            test_metrics['rank']['topk_ndcg'][1], test_metrics['rank']['topk_ndcg'][2], test_metrics['rank']['topk_ndcg'][3],
                            test_metrics['rank']['topk_ndcg'][4], test_metrics['rank']['topk_ndcg'][5])

                # Save model
                if not pretrain:
                    cur_model_wts = copy.deepcopy(self.model.state_dict())
                    self._save_model(cur_model_wts, ei)

                is_best = train_loss < best_loss
                if is_best:
                    best_loss = train_loss
                    logging.info("Epoch {0} - Loss train: {1:.4f}"
                          .format(ei, train_loss))
                    logging.info(test_results)
                    best_model_wts = copy.deepcopy(self.model.state_dict())
        return best_model_wts

    def test(self, loader, criterion, pretrain, model_wts=None, topk=None, atc_level=0):
        # testing with target model, else test with current model
        if model_wts != None:
            self._model_load(model_wts)

        result_dict = self.epoch(loader, criterion=criterion, train=False, pretrain=pretrain)

        if self.params['gpu'] is not "":
            result_dict['label'] = result_dict['label'].cpu()
            result_dict['prediction'] = result_dict['prediction'].cpu()

        test_y_true, test_y_pred, test_loss = result_dict['label'].numpy(), result_dict['prediction'].numpy(), result_dict['loss']
        test_patient, test_drug, test_path = result_dict['patient'].numpy(), result_dict['drug'].numpy(), result_dict['path'].numpy()
        eval_metrics = {}
        eval_metrics['test_loss'] = test_loss
        eval_df = pd.DataFrame({
            'patient': test_patient,
            'drug': test_drug,
            'atc1': test_path[:, 0],
            'atc2': test_path[:, 1],
            'atc3': test_path[:, 2],
            'atc4': test_path[:, 3],
            'atc5': test_path[:, 4],
            'label': test_y_true,
            'prediction': test_y_pred
        })
        if topk is not None:
            eval_metrics['rank'] = ut.rank_eval(eval_df, topk)
        if atc_level > 0:
            eval_metrics['atc_rank'] = ut.atc_eval(eval_df, topk, atc_level)
        return eval_metrics

    def epoch(self, loader, criterion, optimizer=None, train=True, pretrain=False):
        if train and not optimizer:
            raise AttributeError("Optimizer should be given for training")

        if train:
            mode = 'Train'
            self.model.train()
            if not pretrain:
                loader.dataset.ng_sample(self.params['n_negative_sample'])
        else:
            mode = 'Eval'
            self.model.eval()

        losses = AverageMeter()
        labels = []
        outputs = []

        patients = []
        drugs = []
        paths = []

        for bi, batch in enumerate(tqdm(loader, desc="{} batches".format(mode), leave=False, ascii=True)):
            all_patient = batch['patient']
            all_drug = batch['drug']
            all_demo = batch['demo']
            all_se = batch['se']
            all_drug_path = batch['drug_path']
            all_label = batch['label']
            all_ehr = batch['ehr']
            all_ehr_mask = (all_ehr != 0).unsqueeze(-2)

            if self.params['gpu'] is not "":
                all_patient = all_patient.cuda()
                all_drug = all_drug.cuda()
                all_drug_path = all_drug_path.cuda()
                all_label = all_label.cuda()
                all_ehr = all_ehr.cuda()
                all_ehr_mask = all_ehr_mask.cuda()
                all_demo = all_demo.cuda()
                all_se = all_se.cuda()

            health_rep, path_rep, prediction_batch = self.model(all_patient, all_drug, all_demo, all_se,
                                                                all_drug_path, all_ehr, all_ehr_mask)

            if pretrain:
                loss, output_pred = criterion(health_rep, path_rep)
            else:
                loss = criterion(prediction_batch.view(-1), all_label)

            assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'
            labels.append(all_label)
            outputs.append(prediction_batch.view(-1).detach())
            losses.update(loss.item(), all_label.size(0))

            if not train:
                patients.append(torch.squeeze(batch['patient']))
                drugs.append(torch.squeeze(batch['drug']))
                paths.append(batch['drug_path'])

            # compute gradient and do update step
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if train:
            return {
                'prediction': torch.cat(outputs, 0),
                'label': torch.cat(labels, 0),
                'loss': losses.avg,
            }
        else:
            return {
                'prediction': torch.cat(outputs, 0),
                'label': torch.cat(labels, 0),
                'loss': losses.avg,
                # only return when model is tested, info is needed for evaluation.
                'patient': torch.cat(patients, 0),
                'drug': torch.cat(drugs, 0),
                'path': torch.cat(paths, 0)
            }

    def _save_model(self, model_wts, epoch):
        fName = os.path.join('../result/model_saved/', "{}_Model_e-{}".format(datetime.now().strftime("%Y-%m-%d"), epoch))
        torch.save(model_wts, fName + '_params.pth')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(10)
    logger.handlers = [] #clean up existing handler

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(filename)s:%(funcName)s:%(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

def parse_args():
    ap = argparse.ArgumentParser('program')
    ap.add_argument('-f', '--test', default = False, action="store_true",
                    help="Load the models and apply them on the test data to get the predictions.")
    ap.add_argument('-e', '--epochs', required=False, default = 20, type=int, help="setting epoch number.")
    ap.add_argument('--pretrain_epochs', required=False, default=10, type=int, help="setting pretrain epoch number.")
    ap.add_argument("--gpu", type=str, default="", help="0 for gpu card ID, empty means cpu")
    ap.add_argument('--lr', required=False, default=0.001, help="learning rate setting", type=float)
    ap.add_argument('--batch', required=False, default=512, help="batch size", type=int)
    ap.add_argument('--norm', required=False, default=0.001, help="L2 norm on parameters", type=float)
    ap.add_argument('--neg_sample', required=False, default=2, help="negative sample size", type=int)
    ap.add_argument('--dim', required=False, default=64, help="latent dimensions", type=int)
    ap.add_argument('--dropout', required=False, default=0.1, help="dropout probability", type=float)
    ap.add_argument('--bidirectional', required=False, default=False, action="store_true", help="directional for rnn")
    ap.add_argument('--initializer', required=False, default='uniform', help="specify initializer for networks")
    ap.add_argument('--predictor', required=False, default='dot', help="specify predictor for output")
    arg = ap.parse_args()
    return arg

def get_data_info():
    patient_idx = pd.read_csv('../data/patient_index.csv')['patient_idx']
    drug_idx = pd.read_csv('../data/drug_index.csv')['drug_idx']
    atc_idx = pd.read_csv('../data/atc_index.csv')['atc_idx']
    icd_idx = pd.read_csv('../data/icd_index.csv')['icd_idx']
    patient_demo_dim = len(pd.read_csv('../data/patient_demo_code.csv').columns)-1
    drug_se_dim = len(pd.read_csv('../data/drug_se_code.csv').columns)-1
    return {
        'patient': len(patient_idx.unique().tolist()),
        'drug': len(drug_idx.tolist()),
        'atc': len(atc_idx.tolist()),
        'icd': len(icd_idx.tolist()),
        'demo_dim': patient_demo_dim,
        'se_dim': drug_se_dim
    }

if __name__ == "__main__":
    args = parse_args()
    datainfo = get_data_info()
    params = {
        "K": args.dim,
        "n_patient": datainfo['patient'],
        "n_drug": datainfo['drug'],
        "n_atc": datainfo['atc'],
        'n_icd':datainfo['icd'],
        'demo_dim': datainfo['demo_dim'],
        'se_dim': datainfo['se_dim'],
        'n_negative_sample': args.neg_sample,
        'batch_size': args.batch,
        'dropout': args.dropout,
        'epochs': args.epochs,
        'pretrain_epochs': args.pretrain_epochs,
        'L2_norm': args.norm,
        'seed': 48,
        'lr': args.lr,
        'bidirectional': args.bidirectional,
        'initializer': args.initializer,
        'gpu': args.gpu,
        'predictor': args.predictor,
        'transformer_order': True,
        'softmax_temperature': 1.
    }

    train_data_code = pd.read_csv('../data/train_data_code.csv')
    test_data_code = pd.read_csv('../data/test_data_code.csv')
    drug_path_code = pd.read_csv('../data/drug_path_code.csv')
    icd_path_code = pd.read_csv('../data/icd_path_code.csv')

    patient_demo_code = pd.read_csv('../data/patient_demo_code.csv')
    drug_se_code = pd.read_csv('../data/drug_se_code.csv')

    index_data = {}
    index_data['patient_idx_df'] = pd.read_csv('../data/patient_index.csv')
    index_data['drug_idx_df'] = pd.read_csv('../data/drug_index.csv')
    index_data['atc_idx_df'] = pd.read_csv('../data/atc_index.csv')
    index_data['icd_idx_df'] = pd.read_csv('../data/icd_index.csv')

    data = {
        'trainset': train_data_code.values,
        'testset': test_data_code.values,
        'drug_path': drug_path_code.values,
        'icd_path': icd_path_code.values,
        'patient_demo': patient_demo_code.values,
        'drug_se': drug_se_code.values,
        'index_data': index_data
    }

    log_file_name = '../{}.log'.format(datetime.now().strftime("%Y-%m-%d_%H-%M"))
    set_logger(log_file_name)
    logging.info("\n ***************************** \n ")

    model = Pipeline(params, data)
    if not args.test:
        pretrain_model = model.train(pretrain=True)
        model.train(model_params=pretrain_model, pretrain=False)

