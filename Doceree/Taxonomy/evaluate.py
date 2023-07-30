from utils import *
from preprocesing import *
from utils import *
from test_dataset import *


def evaluate(model, test_data):
    fin_outputs = []

    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input in test_dataloader:

              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)
              fin_outputs.extend(output.argmax(dim=1).cpu().detach().numpy().tolist())
    return fin_outputs

if __name__ == '__main__':
    preds_test = evaluate(model,test)