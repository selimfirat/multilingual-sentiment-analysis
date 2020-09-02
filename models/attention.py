from __future__ import print_function, division

import torch
from torch import nn

from torch.autograd import Variable
from torch.nn import Module
from torch.nn.parameter import Parameter

class Attention(Module):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, attention_size, device, return_attention=False):
        """ Initialize the attention layer

        # Arguments:
            attention_size: Size of the attention vector.
            return_attention: If true, output will include the weight for each input token
                              used for the prediction

        """
        super(Attention, self).__init__()
        self.return_attention = return_attention
        self.attention_size = attention_size
        self.device = device
        #self.attention_vector = Parameter(torch.doubleTensor(attention_size))
        self.attention_vector = Parameter(torch.empty(self.attention_size, device=self.device, dtype=torch.double, requires_grad=True))
        self.attention_vector.data.normal_(std=0.05) # Initialize attention vector
        #nn.init.xavier_uniform_(self.attention_vector)

    def __repr__(self):
        s = '{name}({attention_size}, return attention={return_attention})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, input_lengths):
        #print(inputs.size(), self.attention_vector.size())
        """ Forward pass.

        # Arguments:
            inputs (Torch.Variable): Tensor of input sequences
            input_lengths (torch.LongTensor): Lengths of the sequences

        # Return:
            Tuple with (representations and attentions if self.return_attention else None).
        """
        logits = inputs.matmul(self.attention_vector)
        unnorm_ai = (logits - logits.max()).exp()

        # Compute a mask for the attention on the padded sequences
        # See e.g. https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
        #max_len = 1
        max_len = unnorm_ai.shape[1]
        idxes = torch.arange(0, max_len, out=torch.empty(max_len, dtype=torch.long, device=self.device), device=self.device).unsqueeze(0)
        mask = Variable((idxes < input_lengths.unsqueeze(1)).double())

        # apply mask and renormalize attention scores (weights)
        masked_weights = unnorm_ai * mask
        att_sums = masked_weights.sum(dim=1, keepdim=True)  # sums per sequence
        attentions = masked_weights.div(att_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(dim=1)

        return (representations, attentions if self.return_attention else None)