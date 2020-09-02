import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from .attention import Attention
import torch.nn.functional as F

class RecurrentNetwork(nn.Module):

    def __init__(self, cfg, output_size, embedding_size):
        super(RecurrentNetwork, self).__init__()
        self.cfg = cfg
        self.output_size = output_size
        self.embedding_size = embedding_size

        self.dropout = nn.Dropout(self.cfg["dropout"])

        self.num_kernels = 100
        self.kernel_sizes = [2, 3, 4]
        self.convs = torch.nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            self.convs.append(torch.nn.Conv1d((2 if self.cfg["bidirectional"] else 1) * self.cfg["hidden_size"], self.num_kernels,
                kernel_size, padding=kernel_size - 1).double().to(self.cfg["device"]))

        self.init_model()
        self.init_final()
        self.final_ft = None
        self.ft_model = None

        self.labels = ["anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism", "sadness", "surprise", "trust"]
        self.pairs = [("anger", "fear"), ("disgust", "trust"), ("sadness", "joy"), ("surprise", "anticipation"), ("optimism", "pessimism"), ("love", )]

        if self.cfg["pooling"] == "attention":
            att_size = (2 if self.cfg["bidirectional"] else 1) * self.cfg["hidden_size"]
            self.attention_layer = Attention(att_size, self.cfg["device"], return_attention=True)

    def init_final(self):
        in_size = self.num_kernels * len(self.kernel_sizes) if self.cfg["pooling"] == "rcnn" else (2 if self.cfg["bidirectional"] else 1) * self.cfg["hidden_size"]
        if hasattr(self, "ft_model") and self.ft_model is not None:
            if "concat_hidden" in self.cfg["finetune_type"]:
                in_size += self.ft_model.model.hidden_size*2 #(2 if self.cfg["bidirectional"] else 1)*self.cfg["hidden_size"]
            elif "concat_final" in self.cfg["finetune_type"]:
                in_size += self.ft_model.output_size
        self.final = nn.Linear(in_size, self.output_size).double().to(self.cfg["device"])

    def add_finetune_layer(self, output_size):

        self.init_final()
        self.final_ft = nn.Linear(self.output_size, output_size).double().to(self.cfg["device"])
        self.sigmoid_ft = nn.Sigmoid().to(self.cfg["device"])
        self.output_size = output_size

    def init_model(self):

        if self.cfg["model"] == "lstm":
            self.model = nn.LSTM(self.embedding_size, hidden_size=self.cfg["hidden_size"], num_layers=self.cfg["num_layers"], bidirectional=self.cfg["bidirectional"], batch_first=True, dropout=self.cfg["rec_dropout"])
        elif self.cfg["model"] == "gru":
            self.model = nn.GRU(self.embedding_size, hidden_size=self.cfg["hidden_size"], num_layers=self.cfg["num_layers"], bidirectional=self.cfg["bidirectional"], batch_first=True, dropout=self.cfg["rec_dropout"])
        elif self.cfg["model"] == "rnn":
            self.model = nn.RNN(self.embedding_size, hidden_size=self.cfg["hidden_size"], num_layers=self.cfg["num_layers"], bidirectional=self.cfg["bidirectional"], batch_first=True, dropout=self.cfg["rec_dropout"])
        else:
            raise Exception("Unknown recurrent model.")

        self.model = self.model.double().to(self.cfg["device"])

    def init_hidden(self, batch_size):

        hs = self.cfg["num_layers"]*(2 if self.cfg["bidirectional"] else 1)
        h = torch.empty(hs, batch_size, self.cfg["hidden_size"], dtype=torch.double).normal_(0, 0.01).to(self.cfg["device"])

        if self.cfg["model"] == "lstm":
            c = torch.empty(hs, batch_size, self.cfg["hidden_size"], dtype=torch.double).normal_(0, 0.01).to(self.cfg["device"])

            return (h, c)

        return h

    def forward_hidden(self, X, len_X, return_out=False):
        batch_size = X.shape[0]
        hidden = self.init_hidden(batch_size)

        packed_X = pack_padded_sequence(X, len_X, batch_first=True, enforce_sorted=False)

        out, hidden = self.model.forward(packed_X, hidden)

        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        if self.cfg["pooling"] == "attention":
            #input_lengths = torch.LongTensor([torch.max(hidden[i, :].data.nonzero()) + 1 for i in range(hidden.size()[0])]).to(self.cfg["device"])

            hidden, att_weights = self.forward_attention(out, len_X)
        elif self.cfg["pooling"] in ["mean", "max", "last"]:
            hidden = torch.empty((batch_size, self.cfg["hidden_size"] * (2 if self.cfg["bidirectional"] else 1)), dtype=out.dtype,
                                device=self.cfg["device"])

            for i in range(batch_size):
                if self.cfg["pooling"] == "mean":
                    hidden[i, :] = torch.mean(out[i, :len_X[i], :], dim=0)
                elif self.cfg["pooling"] == "max":
                    hidden[i, :] = torch.max(out[i, :len_X[i], :], dim=0)[0]
                elif self.cfg["pooling"] == "last":
                    hidden[i, :] = out[i, len_X[i] - 1, :]
        elif self.cfg["pooling"] == "rcnn":
            doc_embedding = out.transpose(1, 2)
            pooled_outputs = []
            for _, conv in enumerate(self.convs):
                convolution = F.relu(conv(doc_embedding))
                pooled = torch.topk(convolution, 1)[0].view(
                    convolution.size(0), -1)
                pooled_outputs.append(pooled)

            hidden = torch.cat(pooled_outputs, 1)

        hidden = self.dropout(hidden)

        return hidden

    def forward_attention(self, input_seqs, input_lengths):
        return self.attention_layer.forward(input_seqs, input_lengths)

    def forward(self, X, len_X, return_hidden=False):

        hidden = self.forward_hidden(X, len_X)
        if hasattr(self, "ft_model") and self.ft_model is not None and "concat_final" in self.cfg["finetune_type"]:
            if "freezed" in self.cfg["finetune_type"]:
                self.ft_model.eval()
            ft_hidden = self.ft_model.forward(X, len_X)
            hidden = torch.cat([ft_hidden, hidden], dim=1)
        elif hasattr(self, "ft_model") and self.ft_model is not None and "concat_hidden" in self.cfg["finetune_type"]:
            if "freezed" in self.cfg["finetune_type"]:
                self.ft_model.eval()
            ft_hidden = self.ft_model.forward_hidden(X, len_X)
            hidden = torch.cat([hidden, ft_hidden], dim=1)

        final = self.final.forward(hidden)

        if self.final_ft is not None:
            final = self.sigmoid_ft(final)
            final = self.final_ft(final)

        if return_hidden:
            return final, hidden
        else:
            return final
