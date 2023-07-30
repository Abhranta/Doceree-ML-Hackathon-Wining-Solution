from utils import *
class Preprocess():
    def __init__(self):
        train_path = "/kaggle/input/machine-learning-hackathon-ii-by-doceree-media/DOCREE-FILES-FINALE-28thJULY/DOCREE_FILES_FINALE_28thJULY/Doceree-HCP_Train (2).csv"
        test_path = "/kaggle/input/machine-learning-hackathon-ii-by-doceree-media/DOCREE-FILES-FINALE-28thJULY/DOCREE_FILES_FINALE_28thJULY/Doceree-HCP_Test.csv"
        sub_path = "/kaggle/input/machine-learning-hackathon-ii-by-doceree-media/DOCREE-FILES-FINALE-28thJULY/DOCREE_FILES_FINALE_28thJULY/DOCREE_SAMPLE_SUBMISSION.csv"



        df_train = pd.read_csv(train_path, encoding='latin-1')
        df_test = pd.read_csv(test_path, encoding='latin-1')
        df_sub = pd.read_csv(sub_path, encoding='latin-1')


    def concatText(self , x):
        x = x.split('|')
        x = ' '.join(x)
        return x
    
    def preproc(self):

        self.df_train['KEYWORDS'] = self.df_train['KEYWORDS'].apply(self.concatText)
        self.df_test['KEYWORDS'] = self.df_test['KEYWORDS'].apply(self.concatText)

        self.df_train['USERAGENT'] = self.df_train['USERAGENT'].fillna('') 
        self.df_test['USERAGENT'] = self.df_test['USERAGENT'].fillna('') 

        self.df_train['Total'] = self.df_train['USERAGENT'] + ' ' + self.df_train['KEYWORDS'] + ' ' +self.df_train['URL'] 
        self.df_test['Total'] = self.df_test['USERAGENT'] + ' ' + self.df_test['KEYWORDS'] +' ' + self.df_test['URL'] 

        train = self.df_train[["Total", "TAXONOMY"]]

        test = self.df_test[["Total"]]

        le_TAXONOMY = LabelEncoder()
        train['TAXONOMY'] = le_TAXONOMY.fit_transform(train['TAXONOMY'])
        return train , test

