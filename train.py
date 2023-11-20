#source: https://teddylee777.github.io/huggingface/bert-kor-text-classification/
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#eliminate warnings
os.environ["TOKENIZERS_PARALLELISM"] = 'true'
#importing data 
import pandas as pd
train = pd.read_csv('./data/ratings_train.txt', sep='\t')
test = pd.read_csv('./data/ratings_test.txt', sep='\t')
#set parameters
BATCH_SIZE = 12
lr = 1e-6
num_epochs = 10
divide = 2
data_size = train.shape[0]//divide
portion = 1/divide
model_name = f'b:{BATCH_SIZE},e:{num_epochs},s:{data_size},lr:{lr}'
train['length'] = train['document'].apply(lambda x: len(str(x)))
test['length'] = test['document'].apply(lambda x: len(str(x)))
train = train.loc[train['length'] > 5] #getting more than 5 length of sentences

train = train.sample(data_size)
test = test.loc[test['length'] > 5]
test = test.sample(data_size//5)

#set model checkpoint
CHECKPOINT_NAME = 'kykim/bert-kor-base'
import torch
from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
#creating dataset
class TokenDataset(Dataset):
  
    def __init__(self, dataframe, tokenizer_pretrained):
        #sentence, label 
        self.data = dataframe        
        #BERT tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_pretrained)
  
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):
        sentence = self.data.iloc[idx]['document']
        label = self.data.iloc[idx]['label']

        tokens = self.tokenizer(
            sentence,                #one sentence
            return_tensors='pt',     #to pytorch tensor
            truncation=True,         
            padding='max_length',    
            add_special_tokens=True 
        )

        input_ids = tokens['input_ids'].squeeze(0)           # 2D -> 1D
        attention_mask = tokens['attention_mask'].squeeze(0) # 2D -> 1D
        token_type_ids = torch.zeros_like(attention_mask)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask, 
            'token_type_ids': token_type_ids,
        }, torch.tensor(label)
#set tokenizer
tokenizer_pretrained = CHECKPOINT_NAME
#test and train dataset
train_data = TokenDataset(train, tokenizer_pretrained)
test_data = TokenDataset(test, tokenizer_pretrained)
#set dataloader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
#check dataloader
inputs, labels = next(iter(train_loader))
#check input keys
#print(inputs.keys())
#print(inputs['input_ids'].shape) #maximum input size is 512 vecotrs for kobert
#import model
from transformers import BertModel
model_bert = BertModel.from_pretrained(CHECKPOINT_NAME) #stored in pytorch dir
#check model
#print(model_bert.config)
#check model forwarding 
#outputs = model_bert(**inputs)
#print(outputs.keys())
#creating abstract model for fine tuning
class CustomBertModel(nn.Module):
    def __init__(self, bert_pretrained, dropout_rate=0.5):
        super(CustomBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_pretrained)
        self.dr = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(768, 2)
    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = output['last_hidden_state']
        x = self.dr(last_hidden_state[:, 0, :])
        x = self.fc(x)
        return x
#set model
bert = CustomBertModel(CHECKPOINT_NAME)

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

#set device
kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs])
device = accelerator.device
bert = bert.to(device)

#loss 정의: CrossEntropyLoss
loss_fn = nn.CrossEntropyLoss()

# 옵티마이저 정의: bert.paramters()와 learning_rate 설정
optimizer = optim.AdamW(bert.parameters(), lr=lr)

train_loader, test_loader, loss_fn, optimizer, bert = accelerator.prepare(train_loader, test_loader, loss_fn, optimizer, bert)

from tqdm import tqdm #progress bar
def model_train(model, data_loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0
    corr = 0
    counts = 0
    
    prograss_bar = tqdm(data_loader, unit='batch', total=len(data_loader), mininterval=1)
    
    for idx, (inputs, labels) in enumerate(prograss_bar):
        inputs = {k:v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        output = model(**inputs)
        
        loss = loss_fn(output, labels)
        
        accelerator.backward(loss)
        
        optimizer.step()
        
        # output의 max(dim=1)은 max probability와 max index를 반환합니다.
        # max probability는 무시하고, max index는 pred에 저장하여 label 값과 대조하여 정확도를 도출합니다.
        _, pred = output.max(dim=1)
        pred = accelerator.gather(pred)
        labels = accelerator.gather(labels)
        # pred.eq(lbl).sum() 은 정확히 맞춘 label의 합계를 계산합니다. item()은 tensor에서 값을 추출합니다.
        # 합계는 corr 변수에 누적합니다.
        if accelerator.is_local_main_process:
            
            corr += pred.eq(labels).sum().item()

            counts += len(labels)

        # loss 값은 1개 배치의 평균 손실(loss) 입니다. img.size(0)은 배치사이즈(batch size) 입니다.
        # loss 와 img.size(0)를 곱하면 1개 배치의 전체 loss가 계산됩니다.
        # 이를 누적한 뒤 Epoch 종료시 전체 데이터셋의 개수로 나누어 평균 loss를 산출합니다.
            running_loss += loss.item() * labels.size(0)

        #update progress bar

            prograss_bar.set_description(f"training loss: {running_loss/(idx+1):.5f}, training accuracy: {corr / counts:.5f}")
    
    #accuracy
    acc = corr / len(data_loader.dataset) #this part must be divided by number of processes
    
    # train_loss, train_acc
    return running_loss / len(data_loader.dataset), acc #gathering results from all processes

def model_evaluate(model, data_loader, loss_fn, device):
    # model.eval()은 모델을 평가모드로 설정을 바꾸어 줍니다. 
    # dropout과 같은 layer의 역할 변경을 위하여 evaluation 진행시 꼭 필요한 절차 입니다.
    model.eval()
    
    # Gradient가 업데이트 되는 것을 방지 하기 위하여 반드시 필요합니다.
    with torch.no_grad():
        # loss와 accuracy 계산을 위한 임시 변수 입니다. 0으로 초기화합니다.
        corr = 0
        running_loss = 0
        
        # 배치별 evaluation을 진행합니다.
        for inputs, labels in data_loader:
            # inputs, label 데이터를 device 에 올립니다. (cuda:0 혹은 cpu)
            inputs = {k:v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            
            # 모델에 Forward Propagation을 하여 결과를 도출합니다.
            output = model(**inputs)
            # output의 max(dim=1)은 max probability와 max index를 반환합니다.
            # max probability는 무시하고, max index는 pred에 저장하여 label 값과 대조하여 정확도를 도출합니다.
            _, pred = output.max(dim=1)
            pred = accelerator.gather(pred)
            labels = accelerator.gather(labels)
            output = accelerator.gather(output)
            # pred.eq(lbl).sum() 은 정확히 맞춘 label의 합계를 계산합니다. item()은 tensor에서 값을 추출합니다.
            # 합계는 corr 변수에 누적합니다.
            if accelerator.is_local_main_process:
                corr += torch.sum(pred.eq(labels)).item()
            # loss 값은 1개 배치의 평균 손실(loss) 입니다. img.size(0)은 배치사이즈(batch size) 입니다.
            # loss 와 img.size(0)를 곱하면 1개 배치의 전체 loss가 계산됩니다.
            # 이를 누적한 뒤 Epoch 종료시 전체 데이터셋의 개수로 나누어 평균 loss를 산출합니다.
                running_loss += loss_fn(output, labels).item() * labels.size(0)
        # validation 정확도를 계산합니다.
        # 누적한 정답숫자를 전체 데이터셋의 숫자로 나누어 최종 accuracy를 산출합니다.
        acc = corr / len(data_loader.dataset)
        
        # 결과를 반환합니다.
        # val_loss, val_acc
        return running_loss / len(data_loader.dataset), acc #gathering results from all processes

def main(lr = lr, batch_size = BATCH_SIZE, num_epochs = num_epochs, model_name = model_name, portion = portion):
    #logging training process
    #import wandb

    # 최대 Epoch을 지정합니다.

    # checkpoint로 저장할 모델의 이름을 정의 합니다.

    min_loss = np.inf
    """
    if accelerator.is_local_main_process:
        wandb.init(
        # set the wandb project where this run will be logged
        project="LIS8040 BERT Sentiment Analysis",

        # track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "batch_size": batch_size * accelerator.num_processes,
        "architecture": "KoBERT",
        "dataset": "Naver Sentiment Movie Corpus",
        "epochs": num_epochs,
        "portion": portion,
        })
    """
    # Epoch 별 훈련 및 검증을 수행합니다.
    for epoch in range(num_epochs):
        # Model Training
        # 훈련 손실과 정확도를 반환 받습니다.
        train_loss, train_acc = model_train(bert, train_loader, loss_fn, optimizer, device)
        #train_loss, train_acc = (0,0)
        # 검증 손실과 검증 정확도를 반환 받습니다.
        val_loss, val_acc = model_evaluate(bert, test_loader, loss_fn, device)   

        # val_loss 가 개선되었다면 min_loss를 갱신하고 model의 가중치(weights)를 저장합니다.
        if val_loss < min_loss:
            accelerator.print(f'[INFO] val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. Saving Model!')
            min_loss = val_loss
            accelerator.save_model(bert, f"/scratch/paneah/check_point/{model_name}")
        
        # Epoch 별 결과를 출력합니다.
        if accelerator.is_local_main_process:
            print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')
            #wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
if __name__ == "__main__": 
    main()
