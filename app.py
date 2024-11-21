import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pickle
import streamlit as st

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the VGG16-based feature extractor
class VGG16FeatureExtractor(nn.Module):
    def __init__(self, embed_size=256):
        super(VGG16FeatureExtractor, self).__init__()
        from torchvision import models
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool
        self.fc = nn.Linear(512 * 7 * 7, embed_size)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Define the caption generator model
class CaptionGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(CaptionGenerator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.reduce_dim = nn.Linear(embed_size * 2, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions=None):
        if captions is not None:
            captions = captions.long()  # Ensure captions are LongTensor
            embeddings = self.embed(captions)
            features = features.unsqueeze(1).expand(-1, embeddings.size(1), -1)
            inputs = torch.cat((features, embeddings), dim=2)
            reduced_inputs = self.reduce_dim(inputs)
            lstm_out, _ = self.lstm(reduced_inputs)
            outputs = self.linear(lstm_out)
        else:
            inputs = features.unsqueeze(1)
            states = None
            outputs = []
            for _ in range(20):
                lstm_out, states = self.lstm(inputs, states)
                out = self.linear(lstm_out.squeeze(1))
                predicted = out.argmax(dim=1)
                outputs.append(predicted.item())
                inputs = self.embed(predicted.long()).unsqueeze(1)
            return outputs
        return outputs

# Load model and vocabulary
@st.cache(allow_output_mutation=True)
def load_model_vocab():
    feature_extractor = VGG16FeatureExtractor(embed_size=256).to(device)
    feature_extractor.eval()

    model = CaptionGenerator(vocab_size=8922, embed_size=256, hidden_size=512, num_layers=1).to(device)
    model.load_state_dict(torch.load('caption_model.pth', map_location=device))
    model.eval()

    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    return feature_extractor, model, vocab

feature_extractor, model, vocab = load_model_vocab()

def indices_to_words(caption_indices, vocab):
    """
    Convert a list of caption indices to their respective words in the vocabulary.

    Args:
        caption_indices (list): List of integers representing the caption indices.
        vocab (dict): Vocabulary mapping words to indices.

    Returns:
        list: List of words corresponding to the indices.
    """
    # Create a reverse mapping from indices to words
    reverse_vocab = {v: k for k, v in vocab.items()}
    
    # Map each index to the respective word
    words = [reverse_vocab.get(idx, '<unk>') for idx in caption_indices]
    
    return words


st.title("Image Caption Generator")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    def generate_caption(image, feature_extractor, model, vocab):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = feature_extractor(image_tensor)
            caption_indices = model(features)  # List of indices
            words = indices_to_words(caption_indices, vocab)  # Convert indices to words
            caption = []
            for word in words:
                caption.append(word)
                if word == '<end>':  # Stop at the end token
                    break
            return ' '.join(caption)



    caption = generate_caption(image, feature_extractor, model, vocab)
    st.write("Generated Caption:", caption)
