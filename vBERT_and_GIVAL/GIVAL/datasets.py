import torch.utils.data as Data


class MyDataSet_freq(Data.Dataset):
    def __init__(self, inputs):
        super(MyDataSet_freq, self).__init__()
        self.inputs = inputs

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx]


class MyDataSet_label(Data.Dataset):
    def __init__(self, inputs, labels):
        super(MyDataSet_label, self).__init__()
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]
