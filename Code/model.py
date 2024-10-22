import torch
import torch.nn as nn
import torch.nn.functional as F


class ParserModel(nn.Module):


    def __init__(self, embedding_words, embedding_tags, embedding_labels, embed_size = 50, n_words = 18, n_tags = 18, n_labels = 12, hidden_size = 500, n_classes = 80, dropout_prob = 0.5):
        super(ParserModel, self).__init__()
        self.embed_size = embed_size
        self.n_words = n_words
        self.n_tags = n_tags
        self.n_labels = n_labels
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob

        #Embedding Layers
        self.embedding_w = nn.Embedding(embedding_words.shape[0], self.embed_size)
        self.embedding_w.weight = nn.Parameter(torch.tensor(embedding_words))
        self.embedding_t = nn.Embedding(embedding_tags.shape[0], self.embed_size)
        self.embedding_t.weight = nn.Parameter(torch.tensor(embedding_tags))
        self.embedding_l = nn.Embedding(embedding_labels.shape[0], self.embed_size)
        self.embedding_l.weight = nn.Parameter(torch.tensor(embedding_labels))

        #Input layers for words, tags and labels
        self.input_layer_w = nn.Linear(self.n_words * self.embed_size, self.hidden_size)
        self.input_layer_t = nn.Linear(self.n_tags * self.embed_size, self.hidden_size)
        self.input_layer_l = nn.Linear(self.n_labels * self.embed_size, self.hidden_size)

        #Final Output Layer
        self.output_layer = nn.Linear(self.hidden_size, self.n_classes)
        self.dropout = nn.Dropout(self.dropout_prob)

        #Initialization
        nn.init.xavier_uniform_(self.input_layer_l.weight)
        nn.init.xavier_uniform_(self.input_layer_t.weight)
        nn.init.xavier_uniform_(self.input_layer_w.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def get_embeddings(self, w_features, t_features, l_features):
        w_embeddings = self.embedding_w(w_features)
        t_embeddings = self.embedding_t(t_features)
        l_embeddings = self.embedding_l(l_features)
        return w_embeddings, t_embeddings, l_embeddings

    def forward(self, w_features, t_features, l_features):
        #Get Embeddings for words, tags and labels
        w_features_embed, t_features_embed, l_features_embed = self.get_embeddings(w_features, t_features, l_features)
        hidden_w = self.input_layer_w(w_features_embed.view(w_features_embed.size(0), -1))
        hidden_t = self.input_layer_t(t_features_embed.view(t_features_embed.size(0), -1))
        hidden_l = self.input_layer_l(l_features_embed.view(l_features_embed.size(0), -1))
        hidden = hidden_w + hidden_t + hidden_l

        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
        output = self.output_layer(hidden)
        return output


if __name__ == "__main__":
    pass