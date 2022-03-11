import torch
import torch.nn as nn

class ModularModel(nn.Module):
    def __init__(
      self, 
      detection_module, #clickbait_detection_module
      editing_module, #clickbait_editing_module
    ):
      super().__init__()
      self.detection_module = detection_module
      self.editing_module = editing_module

      self.token_sm = nn.Softmax(dim=2) # token softmax
      self.time_sm = nn.Softmax(dim=1) # tme softmax
      self.tok_threshold = nn.Threshold(
        -10000.0, #ARGS.zero_threshold
        0.0) # -10000.0 if ARGS.sequence_softmax else 0.0

    def run_detector(self, pre_id, pre_mask, rel_ids=None, pos_ids=None, categories=None):
      '''Detects Clickbait'''
      _, tok_logits = self.detection_module(
        pre_id, attention_mask=1.0 - pre_mask, rel_ids=rel_ids,
        pos_ids=pos_ids, categories=categories)

    def forward(self, x):
        pass