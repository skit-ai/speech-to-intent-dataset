import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC, HubertModel, HubertForCTC
import whisper

class WhisperModel(nn.Module):
    def __init__(self, model_type="small.en", n_class=14):
        super().__init__()
        self.encoder = whisper.load_model(model_type).encoder

        for param in self.encoder.parameters():
            param.requires_grad = True

        feature_dim = 768 
        # 512 = tiny.en, 
        # 768 = small.en

        self.intent_classifier = nn.Sequential(
            nn.Linear(feature_dim, n_class)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.mean(x, dim=1)
        intent = self.intent_classifier(x)
        return intent

class Wav2VecModel(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")

        for param in self.encoder.parameters():
            param.requires_grad = False
        
        for param in self.encoder.encoder.parameters():
            param.requires_grad = True

        self.intent_classifier = nn.Sequential(
            nn.Linear(1024, 14),
        )

    def forward(self, x):
        x = self.processor(x, sampling_rate=16000, return_tensors="pt")["input_values"].squeeze(0).to("cuda")
        x = self.encoder(x).last_hidden_state
        x = torch.mean(x, dim=1)
        logits = self.intent_classifier(x)
        return logits

class HubertSSLModel(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.encoder = HubertModel.from_pretrained("facebook/hubert-large-ll60k")

        for param in self.encoder.parameters():
            param.requires_grad = False
        
        for param in self.encoder.encoder.parameters():
            param.requires_grad = True

        self.intent_classifier = nn.Sequential(
            nn.Linear(1024, 14),
        )

    def forward(self, x):
        x = self.processor(x, sampling_rate=16000, return_tensors="pt")["input_values"].squeeze(0).to("cuda")
        x = self.encoder(x).last_hidden_state
        x = torch.mean(x, dim=1)
        logits = self.intent_classifier(x)
        return logits