import torch
from torch import nn
from transformers import BertModel


class CustomBert(nn.Module):
    def __init__(
        self,
        concatenate_title: bool = True,
        mode: str = "finetune",
        bert_model="bert-base-uncased",
        local_files_only: bool = False,
        hidden_size: int = 768,
        dropout_prob: float = 0.1,
        num_attention_heads: int = 12,
        num_transformer_layers: int = 12,
    ):
        super(CustomBert, self).__init__()

        self.concatenate_title = concatenate_title
        # TODO: replace with corresponding parameters
        self.bert = BertModel.from_pretrained(
            bert_model,
            local_files_only=local_files_only,
            attention_probs_dropout_prob=dropout_prob,
            hidden_dropout_prob=dropout_prob,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_transformer_layers,
            return_dict=True,
        )

        if mode == "pretrain":
            self.requires_grad_(False)
        elif mode == "finetune":
            self.requires_grad_(True)
        else:
            raise AttributeError("Incorrect mode for BERT model. Should be either 'pretrain' or 'finetune'.")

        input_dim = 4 * hidden_size if concatenate_title else 2 * hidden_size
        self.embedding_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
        )

        # Decision layer
        self.classifier = nn.Linear(hidden_size // 4, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        paper_text: dict,
        reference_text: dict,
        # reference_features: torch.Tensor,
        paper_title: dict | None = None,
        reference_title: dict | None = None,
        return_probabilities: bool = True,
    ):
        if self.concatenate_title and (paper_title is None or reference_title is None):
            raise ValueError("Titles are expected but not provided.")
        if not self.concatenate_title and (paper_title is not None or reference_title is not None):
            raise ValueError("Titles are not expected but were provided.")

        bert_paper_text = self.bert(**paper_text).last_hidden_state[:, 0, :]
        bert_reference_text = self.bert(**reference_text).last_hidden_state[:, 0, :]
        embeddings = torch.cat((bert_paper_text, bert_reference_text), dim=1)  # , reference_features

        if self.concatenate_title:
            bert_paper_title = self.bert(**paper_title).last_hidden_state[:, 0, :]
            bert_reference_title = self.bert(**reference_title).last_hidden_state[:, 0, :]
            embeddings = torch.cat(
                (embeddings, bert_paper_title, bert_reference_title),
                dim=1,
            )

        processed_embeddings = self.embedding_processor(embeddings)
        logits = self.classifier(processed_embeddings)
        return self.softmax(logits) if return_probabilities else logits

    def triplet(
        self,
        anchor_text: dict,
        positive_text: dict,
        negative_text: dict,
        anchor_title: dict | None = None,
        positive_title: dict | None = None,
        negative_title: dict | None = None,
    ):
        if self.concatenate_title and (anchor_title is None or positive_title is None or negative_title is None):
            raise ValueError("Titles are expected but not provided.")
        if not self.concatenate_title and (
            anchor_title is not None or positive_title is not None or negative_title is not None
        ):
            raise ValueError("Titles are not expected but were provided.")

        bert_anchor_text = self.bert(**anchor_text).last_hidden_state[:, 0, :]
        bert_positive_text = self.bert(**positive_text).last_hidden_state[:, 0, :]
        bert_negative_text = self.bert(**negative_text).last_hidden_state[:, 0, :]

        if self.concatenate_title:
            bert_anchor_title = self.bert(**anchor_title).last_hidden_state[:, 0, :]
            bert_positive_title = self.bert(**positive_title).last_hidden_state[:, 0, :]
            bert_negative_title = self.bert(**negative_title).last_hidden_state[:, 0, :]

            bert_anchor_text = torch.cat((bert_anchor_text, bert_anchor_title), dim=1)
            bert_positive_text = torch.cat((bert_positive_text, bert_positive_title), dim=1)
            bert_negative_text = torch.cat((bert_negative_text, bert_negative_title), dim=1)

        return bert_anchor_text, bert_positive_text, bert_negative_text
