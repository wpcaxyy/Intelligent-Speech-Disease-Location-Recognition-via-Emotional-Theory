import torch
from torch.utils.data.dataset import Dataset

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


class IEMOCAPDataset(Dataset):
    def __init__(self, audio_data, audio_label):
        super(IEMOCAPDataset, self).__init__()

        self.audio = torch.Tensor(audio_data)
        self.labels = torch.Tensor(audio_label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        audio_data = (index, self.audio[index])
        audio_data_label = self.labels[index]

        return audio_data, audio_data_label

class RAVDESSDataset(Dataset):
    def __init__(self, audio_data, audio_label):
        super(RAVDESSDataset, self).__init__()

        self.audio = torch.Tensor(audio_data)
        self.labels = torch.Tensor(audio_label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        audio_data = (index, self.audio[index])
        audio_data_label = self.labels[index]

        return audio_data, audio_data_label

class EMODBDataset(Dataset):
    def __init__(self, audio_data, audio_label):
        super(EMODBDataset, self).__init__()

        self.audio = torch.Tensor(audio_data)
        self.labels = torch.Tensor(audio_label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        audio_data = (index, self.audio[index])
        audio_data_label = self.labels[index]

        return audio_data, audio_data_label
