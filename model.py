import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()
        
    def init_weights(self):
        "initialize weights"
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.embed.bias.data.fill_(0)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        print("Init RNN")
        super(DecoderRNN, self).__init__()
        
        # process the args input
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # input: embed_size <- features, a word of captions, 
        # hidden layer: hidden_size
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embed_size)
        self.rnn = nn.LSTM(self.embed_size, self.hidden_size, batch_first=True)
        self.hidden2vocab = nn.Linear(self.hidden_size, self.vocab_size)
        self.init_weights()
        
    def init_weights(self):
        "initialize weights"
        self.word_embeddings.weight.data.uniform_(-0.1, 0.1)
        self.hidden2vocab.weight.data.uniform_(-0.1, 0.1)
        self.hidden2vocab.bias.data.fill_(0)
        
        
    def forward(self, features, captions):
        batch_size = captions.shape[0]
        #Initialize Hidden States
        hiddens = ( torch.zeros(1, batch_size, self.hidden_size).cuda() ,) * 2 #features.shape[0] 
        
        #Embeding captions 
        embeddings = self.word_embeddings(captions)
        
        # Concatenates features and caption embeddings
        input = torch.cat((features.unsqueeze(1), embeddings),1)
       
        # Remove the last word <end> from embedding. 
        indices = torch.tensor(range(captions.shape[1])).cuda()
        input2 = torch.index_select(input, 1, indices)
        
        # LSTM
        outputs, hiddens = self.rnn(input2, hiddens)
        outputs = self.hidden2vocab(outputs)
        
        return outputs
        
       

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        predict_ids = []
        for i in range(max_len):
            hiddens, states = self.rnn(inputs, states)
            
            outputs = self.hidden2vocab(hiddens)
            max_values, max_index = torch.max(outputs[0][0], 0)
            
            predict_ids.append(int(max_index))
            inputs = self.word_embeddings(max_index)
            inputs = inputs.view(1, 1, -1)
            
        return predict_ids