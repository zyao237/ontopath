from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
import pandas as pd
import torch
from tqdm import tqdm
import itertools

class OntopathDataset(Dataset):
    def __init__(self, interact_data, drug_path, icd_path, patient_demo, drug_se, index_data, seed = 101, train=True):
        self.train = train
        random.seed(seed)

        self.patientlist = index_data['patient_idx_df']['patient_idx']
        self.druglist = index_data['drug_idx_df']['drug_idx']
        self.atclist = index_data['atc_idx_df']['atc_idx']
        self.icdlist = index_data['icd_idx_df']['icd_idx']

        drug2path = {}
        for path in drug_path:
            if path[0] not in drug2path:
                drug2path[path[0]] = []
            drug2path[path[0]].append(path[1:])
        self.drug2path = drug2path

        icd2path = {}
        for path in icd_path:
            icd2path[path[0]] = path[::-1]

        patient2demo = {}
        for patient in patient_demo:
            patient2demo[patient[0]] = patient[1:]
        self.patient2demo = patient2demo

        drug2se = {}
        for drug in drug_se:
            drug2se[drug[0]] = drug[1:]
        self.drug2se = drug2se

        if self.train:
            train_data = interact_data[:,[0, 1, 4]]
            drugset = set(self.druglist)
            patient2drug = {}
            for row in train_data:
                if row[0] not in patient2drug:
                    patient2drug[row[0]] = set()
                patient2drug[row[0]].add(row[1])
            self.patient2drug = patient2drug
            patient2negative_drug = {}
            for patient in self.patientlist:
                patient2negative_drug[patient] = drugset - patient2drug[patient]
            self.patient2negative_drug = patient2negative_drug

            self.patient_col = []
            self.drug_col = []

            self.icd_seq_col = []
            for row in train_data:
                self.patient_col.append(row[0])
                self.drug_col.append(row[1])
                self.icd_seq_col.append([icd2path[x] for x in list(map(int, row[2].split(';')))])
            self.drug_path_col = self._sample_pos_drug_path()

            self.patient_all_col = self.patient_col
            self.drug_all_col = self.drug_col
            self.drug_path_all_col = self.drug_path_col
            self.icd_seq_all_col = self.icd_seq_col
            self.label_all_col = [1.] * len(self.patient_col)
        else:
            test_data = interact_data[:,[0, 1, 2, 4]]
            self.patient_col = []
            self.depressant_col = []
            self.depressant_path_col = []
            self.icd_seq_col = []
            self.label_col = []
            depressant_path = {}
            for row in test_data:
                self.patient_col.append(row[0])
                self.depressant_col.append(row[1])
                if row[1] not in depressant_path:
                    depressant_path[row[1]] = random.choice(self.drug2path[row[1]])
                self.depressant_path_col.append(depressant_path[row[1]])
                self.icd_seq_col.append([icd2path[x] for x in list(map(int, row[3].split(';')))])
                self.label_col.append(row[2])

    def ng_sample(self, set_ns_num):
        assert self.train, 'no need to sampling when testing'
        self.drug_path_col = self._sample_pos_drug_path()

        self.patient_ns_col = []
        self.drug_ns_col = []
        self.drug_path_ns_col = []
        self.icd_seq_ns_col = []
        if set_ns_num > 0:
            for patient, icd_seq in tqdm(list(zip(self.patient_col, self.icd_seq_col)), desc="Negative Sampling", leave=False, ascii=True):
                negative_samples = random.sample(self.patient2negative_drug[patient], set_ns_num)
                negative_drug_path = [random.choice(self.drug2path[d]) for d in negative_samples]

                self.patient_ns_col.extend([patient] * set_ns_num)
                self.drug_ns_col.extend(negative_samples)
                self.drug_path_ns_col.extend(negative_drug_path)
                self.icd_seq_ns_col.extend([icd_seq] * set_ns_num)

        self.patient_all_col = self.patient_col + self.patient_ns_col
        self.drug_all_col = self.drug_col + self.drug_ns_col
        self.drug_path_all_col = self.drug_path_col + self.drug_path_ns_col
        self.icd_seq_all_col = self.icd_seq_col + self.icd_seq_ns_col
        self.label_all_col = [1.] * len(self.patient_col) + [0.] * len(self.patient_ns_col)

    def _sample_pos_drug_path(self):
        drug_path_col = []
        for d in self.drug_col:
            drug_path_col.append(random.choice(self.drug2path[d]))
        return drug_path_col

    def __getitem__(self, idx):
        if self.train:
            patient = self.patient_all_col[idx]
            drug = self.drug_all_col[idx]
            drug_path = self.drug_path_all_col[idx]
            icd_seq = self.icd_seq_all_col[idx]
            label = self.label_all_col[idx]
        else:
            patient = self.patient_col[idx]
            drug = self.depressant_col[idx]
            drug_path = self.depressant_path_col[idx]
            icd_seq = self.icd_seq_col[idx]
            label = self.label_col[idx]
        return {
            'patient': patient,
            'drug': drug,
            'drug_path': drug_path,
            'label':label,
            'ehr': icd_seq,
            'demo': self.patient2demo[patient],
            'se': self.drug2se[drug]
        }

    def __len__(self):
        if self.train:
            return len(self.patient_all_col)
        else:
            return len(self.patient_col)

    def get_info(self):
        if self.train:
            samples = len(self.patient_all_col)
        else:
            samples = len(self.patient_col)
        return {
            'patient': len(self.patientlist),
            'drug': len(self.druglist),
            'atc': len(self.atclist),
            'sample': samples
        }

def padding_collate_fn(batch_list):
    patient_batch = []
    drug_batch = []
    drug_path_batch = []
    label_batch = []
    ehr_batch = []
    demo_batch = []
    se_batch = []
    for b in batch_list:
        patient_batch.append(b['patient'])
        drug_batch.append(b['drug'])
        drug_path_batch.append(b['drug_path'])
        label_batch.append(b['label'])
        ehr_batch.append(np.transpose(np.array(b['ehr'])).tolist())
        demo_batch.append(b['demo'])
        se_batch.append(b['se'])

    batch = {}
    batch['patient'] = torch.LongTensor(patient_batch)
    batch['drug'] = torch.LongTensor(drug_batch)
    batch['label'] = torch.FloatTensor(label_batch)
    batch['drug_path'] = torch.LongTensor(drug_path_batch)
    batch['demo'] = torch.FloatTensor(demo_batch)
    batch['se'] = torch.FloatTensor(se_batch)

    ehr_batch = list(itertools.chain.from_iterable(ehr_batch))
    padded_ehr_batch = pad_sequence([torch.LongTensor(x) for x in ehr_batch], batch_first=True)
    batch['ehr'] = padded_ehr_batch

    return batch
