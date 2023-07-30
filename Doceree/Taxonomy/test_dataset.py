from preprocesing import *
from utils import *
from utils import *

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
#         self.labels = df.TAXONOMY.values
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 256, truncation=True,
                                return_tensors="pt") for text in df['Total']]

   # def classes(self):
        #return self.labels

    def __len__(self):
        return len(self.texts)

    #def get_batch_labels(self, idx):
        # Fetch a batch of labels
       # return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        #batch_y = self.get_batch_labels(idx)

        return batch_texts
