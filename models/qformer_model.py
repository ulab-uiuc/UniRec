import torch
import torch.nn as nn
from .qformer import BertModel, BertConfig


class QFormerForItemRepresentation(nn.Module):
    def __init__(self, hidden_size: int = 1024, num_hidden_layers: int = 12, num_attention_heads: int = 16,
                 intermediate_size: int = 4096, num_query_tokens: int = 8, field_embedding_dim: int = 1024,
                 num_fields: int = None, dropout: float = 0.2):
        super().__init__()
        if num_fields is None: raise ValueError("num_fields must be provided")

        self.config = BertConfig(
            hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size, hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
            add_cross_attention=True, query_length=num_query_tokens, encoder_width=field_embedding_dim,
            cross_attention_freq=2
        )
        self.num_query_tokens = num_query_tokens
        self.query_embeddings = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))
        self.qformer = BertModel(self.config, add_pooling_layer=False)
        self.item_representation_head = nn.Linear(hidden_size, field_embedding_dim)
        # Reconstruction head and a projection to match number of fields
        self.reconstruction_head = nn.Linear(hidden_size, field_embedding_dim)
        self.field_projection = nn.Linear(num_query_tokens, num_fields)

    def forward(self, field_embeddings: torch.Tensor, attention_mask: torch.Tensor = None):
        batch_size = field_embeddings.shape[0]
        query_embeds = self.query_embeddings.expand(batch_size, -1, -1)
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, field_embeddings.shape[1], device=field_embeddings.device)

        query_attention_mask = torch.ones(batch_size, self.num_query_tokens, device=field_embeddings.device)

        outputs = self.qformer(
            query_embeds=query_embeds, encoder_hidden_states=field_embeddings,
            encoder_attention_mask=attention_mask, attention_mask=query_attention_mask, return_dict=True
        )
        query_outputs = outputs.last_hidden_state
        item_representation = self.item_representation_head(query_outputs.mean(dim=1))
        
        # Reconstruct each field embedding from the query outputs
        reconstructed_fields_from_queries = self.reconstruction_head(query_outputs)
        reconstructed_fields = self.field_projection(reconstructed_fields_from_queries.transpose(1, 2)).transpose(1, 2)
        
        return {
            'query_outputs': query_outputs,
            'item_representation': item_representation,
            'reconstructed_fields': reconstructed_fields
        }