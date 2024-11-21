# import os
# import pickle
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import models, transforms
# from torch.utils.data import DataLoader, Dataset
# from torch.nn.utils.rnn import pad_sequence
# from PIL import Image
# from nltk.tokenize import word_tokenize
# from collections import Counter
# from tqdm import tqdm

# # Ensure nltk tokenizer works
# import nltk
# nltk.download('punkt')

# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# # Define the VGG16 feature extractor
# class VGG16FeatureExtractor(nn.Module):
#     def __init__(self, embed_size=256):
#         super(VGG16FeatureExtractor, self).__init__()
#         vgg16 = models.vgg16(pretrained=True)
#         self.features = vgg16.features.to(device)
#         self.avgpool = vgg16.avgpool.to(device)
#         self.flatten = nn.Flatten().to(device)
#         self.fc = nn.Linear(25088, embed_size).to(device)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = self.flatten(x)
#         x = self.fc(x)
#         return x

# # Define the caption generator model
# class CaptionGenerator(nn.Module):
#     def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
#         super(CaptionGenerator, self).__init__()
#         self.embed = nn.Embedding(vocab_size, embed_size)
#         self.reduce_dim = nn.Linear(2 * embed_size, embed_size)  # Reduction layer
#         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
#         self.linear = nn.Linear(hidden_size, vocab_size)

#     def forward(self, features, captions):
#         embeddings = self.embed(captions)
#         features = features.unsqueeze(1).expand(-1, embeddings.size(1), -1)
#         inputs = torch.cat((features, embeddings), 2)
#         reduced_inputs = self.reduce_dim(inputs)
#         lstm_out, _ = self.lstm(reduced_inputs)
#         outputs = self.linear(lstm_out)
#         return outputs

# # Image preprocessing transforms
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Dataset class for images
# class ImageFeatureDataset(Dataset):
#     def __init__(self, img_dir, transform):
#         self.img_dir = img_dir
#         self.image_files = [os.path.join(img_dir, file) for file in os.listdir(img_dir)]
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_path = self.image_files[idx]
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image, os.path.basename(img_path).split('.')[0]

# # Load captions and build vocabulary
# def load_captions(filename):
#     with open(filename, 'r') as file:
#         text = file.read()
#     captions = {}
#     for line in text.strip().split('\n'):
#         tokens = line.split(',')
#         img_id = tokens[0].split('.')[0]
#         caption = ' '.join(tokens[1:]).strip()
#         if img_id not in captions:
#             captions[img_id] = []
#         captions[img_id].append(caption)
#     return captions

# def build_vocab(captions):
#     counter = Counter()
#     for caps in captions.values():
#         for cap in caps:
#             counter.update(word_tokenize(cap.lower()))
#     vocab = {word: idx + 1 for idx, word in enumerate(counter)}
#     vocab['<pad>'] = 0
#     vocab['<start>'] = len(vocab)
#     vocab['<end>'] = len(vocab) + 1
#     vocab['<unk>'] = len(vocab) + 2
#     return vocab

# def text_to_seq(caption, vocab):
#     tokens = word_tokenize(caption.lower())
#     sequence = [vocab.get(token, vocab['<unk>']) for token in tokens]
#     return [vocab['<start>']] + sequence + [vocab['<end>']]

# def main():
#     img_dir = './flickr8k/Images/'
#     captions_path = './flickr8k/captions.txt'
#     dataset = ImageFeatureDataset(img_dir, transform)
#     data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
#     extractor = VGG16FeatureExtractor(embed_size=256).eval()
    
#     features = {}
#     for imgs, ids in tqdm(data_loader, desc="Extracting features"):
#         imgs = imgs.to(device)
#         with torch.no_grad():
#             feats = extractor(imgs)
#         for i, img_id in enumerate(ids):
#             features[img_id] = feats[i].cpu().numpy()
    
#     pickle.dump(features, open('image_features.pkl', 'wb'))

#     captions = load_captions(captions_path)
#     vocab = build_vocab(captions)
#     vocab_size = len(vocab)
#     print(vocab_size)
    
    # caption_dataset = [(img_id, cap) for img_id in captions for cap in captions[img_id]]
    # caption_features = [features[img_id] for img_id, _ in caption_dataset if img_id in features]
    # caption_seqs = [text_to_seq(cap, vocab) for _, cap in caption_dataset if _ in features]
    # caption_features = torch.tensor(caption_features, dtype=torch.float)
    # caption_seqs = pad_sequence([torch.tensor(seq) for seq in caption_seqs], batch_first=True, padding_value=vocab['<pad>'])

#     model = CaptionGenerator(vocab_size, 256, 512, 1).to(device)
#     criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

    # epochs = 10
    # for epoch in range(epochs):
    #     model.train()
    #     for i in range(0, len(caption_features), 32):
    #         img_features = caption_features[i:i+32].to(device)
    #         captions = caption_seqs[i:i+32].to(device)
    #         outputs = model(img_features, captions[:, :-1])
    #         loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    # torch.save(model.state_dict(), 'caption_model.pth')

# if __name__ == '__main__':
#     main()

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from nltk.tokenize import word_tokenize
from collections import Counter
from tqdm import tqdm
import nltk
nltk.download('punkt')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class VGG16FeatureExtractor(nn.Module):
    def __init__(self, embed_size=256):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features.to(device)
        self.avgpool = vgg16.avgpool.to(device)
        self.flatten = nn.Flatten().to(device)
        self.fc = nn.Linear(25088, embed_size).to(device)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class CaptionGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ImageFeatureDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.image_files = [os.path.join(img_dir, file) for file in os.listdir(img_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image), os.path.basename(img_path).split('.')[0]

def load_captions(filename):
    with open(filename, 'r') as file:
        text = file.read()
    captions = {}
    for line in text.strip().split('\n'):
        tokens = line.split(',')
        img_id = tokens[0].split('.')[0]
        caption = ' '.join(tokens[1:]).strip()
        if img_id not in captions:
            captions[img_id] = []
        captions[img_id].append(caption)
    return captions

def build_vocab(captions):
    counter = Counter()
    for caps in captions.values():
        for cap in caps:
            counter.update(word_tokenize(cap.lower()))
    vocab = {word: idx + 1 for idx, word in enumerate(counter)}
    vocab['<pad>'] = 0
    vocab['<start>'] = len(vocab)
    vocab['<end>'] = len(vocab) + 1
    vocab['<unk>'] = len(vocab) + 2
    return vocab

def save_vocab(vocab, filename):
    with open(filename, 'wb') as f:
        pickle.dump(vocab, f)
    print("Vocabulary successfully saved to", filename)

def text_to_seq(caption, vocab):
    tokens = word_tokenize(caption.lower())
    sequence = [vocab.get(token, vocab['<unk>']) for token in tokens]
    return [vocab['<start>']] + sequence + [vocab['<end>']]

def main():
    img_dir = './flickr8k/Images/'
    captions_path = './flickr8k/captions.txt'
    vocab_file = 'vocab.pkl'
    dataset = ImageFeatureDataset(img_dir, transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    extractor = VGG16FeatureExtractor(embed_size=256).eval()
    
    captions = load_captions(captions_path)
    if not os.path.exists(vocab_file):
        vocab = build_vocab(captions)
        save_vocab(vocab, vocab_file)
    else:
        with open(vocab_file, 'rb') as f:
            vocab = pickle.load(f)

    features = {}
    for imgs, ids in tqdm(data_loader, desc="Extracting features"):
        imgs = imgs.to(device)
        with torch.no_grad():
            feats = extractor(imgs)
        for i, img_id in enumerate(ids):
            features[img_id] = feats[i].cpu().numpy()
    
    pickle.dump(features, open('image_features.pkl', 'wb'))

    model = CaptionGenerator(len(vocab), 256, 512, 1).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    caption_dataset = [(img_id, cap) for img_id in captions for cap in captions[img_id]]
    caption_features = [features[img_id] for img_id, _ in caption_dataset if img_id in features]
    caption_seqs = [text_to_seq(cap, vocab) for _, cap in caption_dataset if _ in features]

    caption_features = torch.tensor(caption_features, dtype=torch.float)
    caption_seqs = pad_sequence([torch.tensor(seq) for seq in caption_seqs], batch_first=True, padding_value=vocab['<pad>'])

    epochs = 10
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(features), 32):
            img_features = caption_features[i:i+32].to(device)
            captions = caption_seqs[i:i+32].to(device)
            outputs = model(img_features, captions[:, :-1])
            loss = criterion(outputs.reshape(-1, len(vocab)), captions[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    torch.save(model.state_dict(), 'caption_model.pth')

if __name__ == '__main__':
    main()
