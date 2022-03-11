import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertSelfAttention

class DetectionModule(BertPreTrainedModel):
  def __init__(self, config, cls_num_labels=2, tok_num_labels=2, tok2id=None):
    super(DetectionModule, self).__init__(config)

    self.bert = BertModel(config)

    self.cls_dropout = nn.Dropout(config.hidden_dropout_prob)
    self.cls_classifier = nn.Linear(config.hidden_size, cls_num_labels)

    self.tok_dropout = nn.Dropout(config.hidden_dropout_prob)
    self.tok_classifier = nn.Linear(config.hidden_size, tok_num_labels)
    
    self.apply(self.init_bert_weights)
