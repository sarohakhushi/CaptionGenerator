import torch
import torch.nn as nn
from torchvision import models

class VGG16FeatureExtractor(nn.Module):
    def __init__(self, embed_size=256):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(25088, embed_size)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class CaptionGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(CaptionGenerator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.reduce_dim = nn.Linear(2 * embed_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        features = features.unsqueeze(1).expand(-1, embeddings.size(1), -1)
        inputs = torch.cat((features, embeddings), 2)
        reduced_inputs = self.reduce_dim(inputs)
        lstm_out, _ = self.lstm(reduced_inputs)
        outputs = self.linear(lstm_out)
        return outputs
