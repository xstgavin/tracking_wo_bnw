from torch.utils.data import Dataset

from .homebrain_sequence import HomeBrain_Sequence

class HomeBrain_Wrapper(Dataset):
    """A Wrapper for the homebrain class to return multiple sequences."""

    def __init__(self, split, dataloader):
        """Initliazes all subset of the dataset.

        Keyword arguments:
        split -- the split of the dataset to use
        dataloader -- args for the MOT_Sequence dataloader
        """

        #train_sequences = ['Venice-2', 'KITTI-17', 'KITTI-13', 'ADL-Rundle-8', 'ADL-Rundle-6', 'ETH-Pedcross2',
        #                   'ETH-Sunnyday', 'ETH-Bahnhof', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte']

        test_sequences = ['2018-10-01-09-00-22-420620','2018-10-01-09-00-39-420675','2018-10-01-09-02-12-420824',
                            '2018-10-01-09-02-17-420839','2018-10-01-09-03-28-421116','2018-10-01-09-03-35-421138',
                            '2018-10-01-09-05-30-421443','2018-10-01-18-01-12-440249','2018-10-01-18-04-08-440564']
        if "homebrain" == split:
            sequences = test_sequences
        else:
            raise NotImplementedError("Image set: {}".format(split))

        self._data = []

        for s in sequences:
            self._data.append(HomeBrain_Sequence(seq_name=s, **dataloader))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]
