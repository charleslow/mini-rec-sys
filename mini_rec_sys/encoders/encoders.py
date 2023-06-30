"""
Wrappers around various transformer models to define a consistent interface
for encoding texts and using for search.
"""
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
from torch import nn
import torch.nn.functional as F
from typing import List


def model_to_encoder_type(model):
    if isinstance(model, BertEncoder):
        return "bert"
    elif isinstance(model, DistilBertEncoder):
        return "distilbert"
    elif isinstance(model, MiniLmEncoder):
        return "minilm"
    else:
        raise ValueError()


def encoder_type_to_model(encoder_type: str, *args, **kwargs):
    """
    Initialize the transformer model based on encoder_type.
    """
    if encoder_type == "bert":
        return BertEncoder(*args, **kwargs)
    elif encoder_type == "distilbert":
        return DistilBertEncoder(*args, **kwargs)
    elif encoder_type == "minilm":
        return MiniLmEncoder(*args, **kwargs)
    else:
        raise ValueError()


def save_model(model: nn.Module, path):
    torch.save(model.state_dict(), path)


def load_model(path: str, encoder_type: str = "bert", *args, **kwargs):
    model = encoder_type_to_model(
        encoder_type=encoder_type, *args, **kwargs
    )  # init model
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


class BaseBertEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        dim_output: int,
        dim_embed: int,
        max_length: int,
        normalize: bool = True,
    ) -> None:
        """
        model_name: e.g. bert-base-cased
        dim_output: dimension of the output embedding of the model
        dim_embed: add an additional linear layer to coerce the embedding
            dimension to a desired size.
        max_length: maximum sequence length to embed.
        normalize: whether to apply L2 normalization to output layer or not.
        """
        super().__init__()
        self.model_name = model_name
        self.dim_embed = dim_embed
        self.max_length = max_length
        self.normalize = normalize

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        configuration = AutoConfig.from_pretrained(model_name)
        configuration.hidden_dropout_prob = 0.1
        self.encoder = AutoModel.from_pretrained(model_name, config=configuration)
        if dim_output != dim_embed:
            self.linear = nn.Linear(dim_output, dim_embed)
        else:
            self.linear = None

    def forward(self, texts: List[str]):
        """
        Embed a list of texts into a (len(texts), self.dim_embed) embedding tensor.
        """
        tokens = self.tokenize(texts)
        out = self.encoder(**tokens, output_attentions=False)
        out = out.last_hidden_state[:, 0, :]  # Use CLS token
        if self.linear is not None:
            out = self.linear(out)
        if self.normalize:
            out = F.normalize(out, p=2, dim=1)
        return out

    def tokenize(self, texts: List[str]):
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        return tokens


class BertEncoder(BaseBertEncoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            model_name="bert-base-cased",
            dim_output=768,
            *args,
            **kwargs,
        )


class DistilBertEncoder(BaseBertEncoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            model_name="distilbert-base-cased",
            dim_output=768,
            *args,
            **kwargs,
        )


class MiniLmEncoder(BaseBertEncoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            dim_output=384,
            *args,
            **kwargs,
        )
