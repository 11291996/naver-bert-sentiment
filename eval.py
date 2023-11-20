#source: https://teddylee777.github.io/huggingface/bert-kor-text-classification/
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#eliminate warnings
os.environ["TOKENIZERS_PARALLELISM"] = 'true'
#set model checkpoint
CHECKPOINT_NAME = 'kykim/bert-kor-base'
from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
model_bert = BertModel.from_pretrained(CHECKPOINT_NAME) #stored in pytorch dir
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
kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs])
device = accelerator.device
accelerator.print(bert)
bert = bert.to(device)
bert = accelerator.prepare(bert)


#load model
unwrapped_model = accelerator.unwrap_model(bert)
model_name = 'b:32,e:10,s:75000,lr:1e-06'
save_directory = f"/scratch/paneah/check_point/{model_name}"
path_to_checkpoint = os.path.join(save_directory,"pytorch_model.bin")
unwrapped_model.load_state_dict(torch.load(path_to_checkpoint))

class CustomPredictor():
    def __init__(self, model, tokenizer, labels: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.labels = labels
        
    def predict(self, sentence):
        # 토큰화 처리
        tokens = self.tokenizer(
            sentence,                # 1개 문장 
            return_tensors='pt',     # 텐서로 반환
            truncation=True,         # 잘라내기 적용
            padding='max_length',    # 패딩 적용
            add_special_tokens=True  # 스페셜 토큰 적용
        )
        tokens.to(device)
        prediction = self.model(**tokens)
        prediction = F.softmax(prediction, dim=1)
        output = prediction.argmax(dim=1).item()
        prob, result = prediction.max(dim=1)[0].item(), self.labels[output]
        print(f'[{result}]\n확률은: {prob*100:.3f}% 입니다.')

tokenizer = BertTokenizerFast.from_pretrained(CHECKPOINT_NAME)

labels = {0: '부정', 1: '긍정'}

predictor = CustomPredictor(unwrapped_model, tokenizer, labels)

def predict_sentence(predictor):
    input_sentence = input('문장을 입력해 주세요: ') #utf-8 #"안녕하세요"
    predictor.predict(input_sentence)

if __name__ == "__main__": 
    while True & accelerator.is_local_main_process:
        predict_sentence(predictor)