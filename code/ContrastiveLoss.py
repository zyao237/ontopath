import torch
import numpy as np

class ConLoss(torch.nn.Module):
    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(ConLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        initial_mask = diag + l1 + l2
        # mask for patient-patient or drug-drug match
        initial_mask[:self.batch_size, :self.batch_size] = 1.
        initial_mask[self.batch_size:, self.batch_size:] = 1.
        initial_mask[self.batch_size:, :self.batch_size] = 1.
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        return v

    def _cosine_simililarity(self, x, y):
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zis, zjs], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        r_pos = torch.diag(similarity_matrix, self.batch_size)
        positives = r_pos.view(self.batch_size, 1)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (self.batch_size), r_pos
