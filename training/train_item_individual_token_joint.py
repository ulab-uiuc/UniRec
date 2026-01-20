import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, Trainer, TrainingArguments, TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import json
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
import pickle

# Add Q-Former model and utilities (refactored layout)
from models.qformer_utils import QFormerForItemRepresentation, QFormerDataset
from models.item_encoder_pure_value import ItemEncoder

# New: paths for Amazon All Beauty
# TRAIN_DATA_PATH = "data_rec/data/Amazon_All_Beauty_all_train_LRanker.json"
# VAL_DATA_PATH = "data_rec/data/Amazon_All_Beauty_all_test_LRanker.json"
TRAIN_DATA_PATH = "data_rec/data/Amazon_All_Beauty_20_train.json"
VAL_DATA_PATH = "data_rec/data/Amazon_All_Beauty_20_test.json"
ITEM_EMB_PATH = "data_rec/embeddings/all_beauty_item_embedding_qwen3_0.6B.json"
QFORMER_CHECKPOINT_PATH = "qformer_checkpoints_contrastive_2_query_tokens/best_qformer_model.pth"
ITEM_DICT_PATH = "data_rec/dict/All_Beauty_item_triplet_dict.json"
# QFORMER_CACHE_DIR is no longer used for pre-computation, but the underlying QFormerDataset will use its own cache.
FIELD_EMBEDDING_CACHE_DIR = "embedding_cache_contrastive/full"

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def load_qformer_model(checkpoint_path, num_fields, eval_mode: bool = True):
    """Loads a pre-trained Q-Former model from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Initialize the model with the loaded configuration
    model = QFormerForItemRepresentation(
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        num_query_tokens=config.query_length,
        field_embedding_dim=config.encoder_width,
        num_fields=num_fields, # Ensure this is passed correctly
        dropout=config.hidden_dropout_prob
    ).to(device)
    
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    if eval_mode:
        model.eval()
    print(f"✅ Q-Former model loaded from {checkpoint_path} in {'eval' if eval_mode else 'train'} mode.")
    return model


def get_field_order(item_dict: Dict) -> List[str]:
    """Analyzes all unique fields present in the item dictionary."""
    print("Analyzing unique fields from item dictionary...")
    all_fields = set()
    for item_data in item_dict.values():
        all_fields.update(item_data.keys())
    
    # Define a consistent order for fields, excluding item_id
    field_order = sorted([f for f in all_fields if f != 'item_id'])
    print(f"Found {len(field_order)} unique fields.")
    return field_order


# Removed get_qformer_query_tokens_for_items as it's no longer needed for joint training


class MultiModalQwenEmbedding(nn.Module):
    def __init__(self, base_model_name: str, qformer_model: nn.Module, use_lora: bool = True, lora_config: Optional[LoraConfig] = None):
        super().__init__()
        self.device = device
        self.use_lora = use_lora
        self.num_history_items = 10
        self.num_query_tokens_per_item = 2
        self.qformer_model = qformer_model

        # Use Qwen/Qwen3-Embedding-0.6B for backbone
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModel.from_pretrained(
            base_model_name,
            device_map=device,
            torch_dtype=torch.float32
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.hidden_size = self.base_model.config.hidden_size
        
        # No projector needed as Q-Former output and LLM hidden size are both 1024

        # Special tokens for history items, each followed by query tokens
        self.history_tokens = []
        for i in range(self.num_history_items):
            for j in range(self.num_query_tokens_per_item):
                self.history_tokens.append(f"<|history_item_{i}_query_{j}|>")
            
        special_tokens = {"additional_special_tokens": self.history_tokens}
        self.tokenizer.add_special_tokens(special_tokens)
        self.base_model.resize_token_embeddings(len(self.tokenizer))

        if use_lora:
            if lora_config is None:
                lora_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=16, lora_alpha=32, lora_dropout=0.1,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
                    bias="none",
                )
            print("🔧 Applying LoRA to base model...")
            self.base_model = get_peft_model(self.base_model, lora_config)
            self.base_model.print_trainable_parameters()

    def forward(self, input_ids, attention_mask=None, history_field_embeddings=None, history_attention_mask=None):
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if history_field_embeddings is not None:
            history_field_embeddings = history_field_embeddings.to(device)
        if history_attention_mask is not None:
            history_attention_mask = history_attention_mask.to(device)

        text_embeds = self.base_model.get_input_embeddings()(input_ids)
        batch_size, seq_len, hidden_size = text_embeds.shape

        if history_field_embeddings is not None and history_attention_mask is not None:
            # Dynamically generate query tokens using the Q-Former
            bh, num_hist, num_fields, field_dim = history_field_embeddings.shape
            
            # Reshape for Q-Former: (batch * num_hist, num_fields, field_dim)
            history_field_embeddings_reshaped = history_field_embeddings.view(bh * num_hist, num_fields, field_dim)
            history_attention_mask_reshaped = history_attention_mask.view(bh * num_hist, num_fields)
            
            qformer_output = self.qformer_model(history_field_embeddings_reshaped, history_attention_mask_reshaped)
            query_outputs = qformer_output['query_outputs']
            
            # Reshape back: (batch, num_hist, num_query_tokens, hidden_size)
            history_item_query_tokens = query_outputs.view(bh, num_hist, self.num_query_tokens_per_item, hidden_size)

            for i in range(self.num_history_items):
                for j in range(self.num_query_tokens_per_item):
                    token_name = f"<|history_item_{i}_query_{j}|>"
                    token_id = self.tokenizer.convert_tokens_to_ids(token_name)
                    
                    # Inject the query token embeddings directly (no projection)
                    query_embeddings = history_item_query_tokens[:, i, j, :]
                    
                    for batch_idx in range(batch_size):
                        positions = (input_ids[batch_idx] == token_id).nonzero(as_tuple=True)[0]
                        if len(positions) > 0:
                            text_embeds[batch_idx, positions] = query_embeddings[batch_idx]
                            
        outputs = self.base_model(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # Use mean pooling of last_hidden_state as embedding
        last_hidden_state = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
        pooled_output = torch.mean(last_hidden_state, dim=1)
        return pooled_output

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self.tokenizer.save_pretrained(save_directory)
        if self.use_lora:
            self.base_model.save_pretrained(save_directory)
        else:
            torch.save(self.base_model.state_dict(), os.path.join(save_directory, "base_model.bin"))
        
        # Save the Q-Former model state dict
        torch.save(self.qformer_model.state_dict(), os.path.join(save_directory, "qformer_model.bin"))
        
        config = {
            "hidden_size": self.hidden_size,
            "use_lora": self.use_lora,
        }
        with open(os.path.join(save_directory, "model_config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Model saved to {save_directory}")

    def get_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for param in self.parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"Trainable params: {trainable_params:,} || All params: {all_param:,} || "
              f"Trainable%: {100 * trainable_params / all_param:.2f}%")
        print(f"🔄 Jointly training Q-Former and LLM LoRA adapters (no projection)")
        return trainable_params, all_param


class ValidationDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, item_qformer_query_tokens, 
                 item_emb_dim: int, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.item_qformer_query_tokens = item_qformer_query_tokens
        self.item_emb_dim = item_emb_dim
        self.max_length = max_length
        self.num_history_items = 10
        self.num_query_tokens_per_item = 2
        
        # Pre-compute shared zero tensors for memory efficiency
        self.zero_query_tokens = torch.zeros(
            (self.num_history_items, self.num_query_tokens_per_item, self.item_emb_dim),
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        history_items = item.get('history_items', [])
        
        # Dynamically fetch Q-Former query tokens for history items
        history_item_query_tokens = self.zero_query_tokens.clone()
        
        num_items_to_process = min(len(history_items), self.num_history_items)
        
        if num_items_to_process > 0:
            item_ids = history_items[:num_items_to_process]
            query_tokens_list = [
                torch.from_numpy(self.item_qformer_query_tokens.get(item_id, np.zeros((self.num_query_tokens_per_item, self.item_emb_dim))))
                for item_id in item_ids
            ]
            
            if query_tokens_list:
                # Stack and pad if necessary
                stacked_tokens = torch.stack(query_tokens_list).to(torch.float32)
                history_item_query_tokens[:num_items_to_process] = stacked_tokens

        gt_item = item['gt_item']
        candidate_items = item['candidate_items']
        candidate_embeddings = item['candidate_embeddings']
        gt_index = candidate_items.index(gt_item)
        gt_embedding = candidate_embeddings[gt_index]
        negative_embeddings = [emb for i, emb in enumerate(candidate_embeddings) if i != gt_index]
        positive_item_embedding = np.array(gt_embedding, dtype=np.float32)
        negative_item_embeddings = np.array(negative_embeddings, dtype=np.float32)

        if len(history_items_embedding) == 0:
            history_items_embedding = np.zeros((10, 1024), dtype=np.float32)

        problem_text = item.get('problem', '')
        input_text = (f"This is a recommendation task. Based on the user's <|history_items|> item interaction history, "
                      f"<text>{problem_text}</text> predict the next item the user is most likely to interact with.")

        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'history_item_query_tokens': history_item_query_tokens,
            'positive_item_embedding': torch.tensor(positive_item_embedding, dtype=torch.float32),
            'negative_item_embeddings': torch.tensor(negative_item_embeddings, dtype=torch.float32),
        }


class MultiModalDataCollator:
    def __init__(self, tokenizer, max_negatives: int = 10):
        self.tokenizer = tokenizer
        self.max_negatives = max_negatives

    def __call__(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        # New: Handle history field embeddings and mask
        history_field_embeddings = torch.stack([item['history_field_embeddings'] for item in batch])
        history_attention_mask = torch.stack([item['history_attention_mask'] for item in batch])
        
        positive_item_embeddings = torch.stack([item['positive_item_embedding'] for item in batch])

        batch_size = len(batch)
        embedding_dim = batch[0]['negative_item_embeddings'].shape[-1]
        padded_negatives = torch.zeros(batch_size, self.max_negatives, embedding_dim)
        neg_masks = torch.zeros(batch_size, self.max_negatives, dtype=torch.bool)
        for i, item in enumerate(batch):
            neg_embs = item['negative_item_embeddings']
            num_negs = min(len(neg_embs), self.max_negatives)
            padded_negatives[i, :num_negs] = neg_embs[:num_negs]
            neg_masks[i, :num_negs] = True

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'history_field_embeddings': history_field_embeddings,
            'history_attention_mask': history_attention_mask,
            'positive_item_embeddings': positive_item_embeddings,
            'negative_item_embeddings': padded_negatives,
            'negative_masks': neg_masks
        }


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, user_embeddings, positive_item_embeddings, negative_item_embeddings, negative_masks=None):
        batch_size = user_embeddings.size(0)
        user_embeddings = F.normalize(user_embeddings, p=2, dim=-1)
        positive_item_embeddings = F.normalize(positive_item_embeddings, p=2, dim=-1)
        negative_item_embeddings = F.normalize(negative_item_embeddings, p=2, dim=-1)
        pos_sim = torch.sum(user_embeddings * positive_item_embeddings, dim=-1) / self.temperature
        neg_sim = torch.bmm(
            user_embeddings.unsqueeze(1),
            negative_item_embeddings.transpose(-2, -1)
        ).squeeze(1) / self.temperature
        if negative_masks is not None:
            neg_sim = neg_sim.masked_fill(~negative_masks, -1e9)
        losses = []
        for i in range(batch_size):
            if negative_masks is not None:
                valid_neg_sim = neg_sim[i][negative_masks[i]]
            else:
                valid_neg_sim = neg_sim[i]
            all_sim = torch.cat([pos_sim[i:i + 1], valid_neg_sim])
            loss = -pos_sim[i] + torch.logsumexp(all_sim, dim=0)
            losses.append(loss)
        return torch.stack(losses).mean()


class MRREvaluator:
    def __init__(self, model, tokenizer, validation_dataset: ValidationDataset):
        self.model = model
        self.tokenizer = tokenizer
        self.validation_dataset = validation_dataset

    def evaluate_mrr(self, batch_size: int = 32) -> float:
        self.model.eval()
        mrr_scores = []
        val_dataloader = torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._validation_collate_fn
        )
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluating MRR"):
                batch_mrr = self._compute_batch_mrr(batch)
                mrr_scores.extend(batch_mrr)
        return np.mean(mrr_scores)

    def _validation_collate_fn(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        history_field_embeddings = torch.stack([item['history_field_embeddings'] for item in batch])
        history_attention_mask = torch.stack([item['history_attention_mask'] for item in batch])
        positive_item_embeddings = torch.stack([item['positive_item_embedding'] for item in batch])
        negative_item_embeddings = [item['negative_item_embeddings'] for item in batch]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'history_field_embeddings': history_field_embeddings,
            'history_attention_mask': history_attention_mask,
            'positive_item_embeddings': positive_item_embeddings,
            'negative_item_embeddings': negative_item_embeddings,
        }

    def _compute_batch_mrr(self, batch) -> List[float]:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        history_field_embeddings = batch['history_field_embeddings'].to(device)
        history_attention_mask = batch['history_attention_mask'].to(device)
        positive_item_embeddings = batch['positive_item_embeddings'].to(device)

        user_embeddings = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            history_field_embeddings=history_field_embeddings,
            history_attention_mask=history_attention_mask
        )
        user_embeddings = F.normalize(user_embeddings, p=2, dim=-1)
        positive_item_embeddings = F.normalize(positive_item_embeddings, p=2, dim=-1)

        batch_mrr = []
        for i in range(len(user_embeddings)):
            user_emb = user_embeddings[i]
            pos_emb = positive_item_embeddings[i]
            neg_embs = F.normalize(batch['negative_item_embeddings'][i].to(device), p=2, dim=-1)
            all_items = torch.cat([pos_emb.unsqueeze(0), neg_embs], dim=0)
            similarities = torch.matmul(user_emb, all_items.t())
            sorted_indices = torch.argsort(similarities, descending=True)
            pos_rank = (sorted_indices == 0).nonzero(as_tuple=True)[0].item() + 1
            mrr = 1.0 / pos_rank
            batch_mrr.append(mrr)
        return batch_mrr


class BestMRRCallback(TrainerCallback):
    def __init__(self, mrr_evaluator: MRREvaluator, save_dir: str = "./best_model",
                 eval_steps: int = 10, save_strategy: str = "best_only"):
        self.mrr_evaluator = mrr_evaluator
        self.save_dir = save_dir
        self.eval_steps = eval_steps
        self.save_strategy = save_strategy
        self.last_eval_step = 0
        self.best_mrr = -1.0
        if self.save_strategy == "both":
            self.best_save_dir = os.path.join(save_dir, "best_model")
            self.latest_save_dir = os.path.join(save_dir, "latest_model")
            os.makedirs(self.best_save_dir, exist_ok=True)
            os.makedirs(self.latest_save_dir, exist_ok=True)
        else:
            os.makedirs(save_dir, exist_ok=True)

    def on_log(self, args, state, control, model, logs=None, **kwargs):
        if (state.global_step > 0 and state.global_step - self.last_eval_step >= self.eval_steps):
            print(f"\n🔍 Evaluating MRR at step {state.global_step}...")
            mrr_score = self.mrr_evaluator.evaluate_mrr()
            print(f"📊 MRR Score: {mrr_score:.4f}")

            if logs is not None:
                logs["eval_mrr"] = mrr_score

            if self.save_strategy == "best_only":
                if mrr_score > self.best_mrr:
                    old_best = self.best_mrr
                    print(f"🏆 New best MRR! Saving shared projector model to {self.save_dir} "
                          f"(MRR improved from {old_best:.4f} to {mrr_score:.4f})")
                    self.best_mrr = mrr_score
                    model.save_pretrained(self.save_dir)
            elif self.save_strategy == "always":
                print(f"💾 Saving latest shared projector model to {self.save_dir} (MRR: {mrr_score:.4f})")
                model.save_pretrained(self.save_dir)
                if mrr_score > self.best_mrr:
                    print(f"🏆 New best MRR achieved! (improved from {self.best_mrr:.4f} to {mrr_score:.4f})")
                    old_best = self.best_mrr
                    self.best_mrr = mrr_score
            elif self.save_strategy == "both":
                print(f"💾 Saving latest shared projector model to {self.latest_save_dir} (MRR: {mrr_score:.4f})")
                model.save_pretrained(self.latest_save_dir)
                if mrr_score > self.best_mrr:
                    old_best = self.best_mrr
                    print(f"🏆 New best MRR! Saving best shared projector model to {self.best_save_dir} "
                          f"(MRR improved from {old_best:.4f} to {mrr_score:.4f})")
                    self.best_mrr = mrr_score
                    model.save_pretrained(self.best_save_dir)
                else:
                    print(f"📊 MRR {mrr_score:.4f} did not improve from best {self.best_mrr:.4f}. Best model not updated.")
            model.train()
            self.last_eval_step = state.global_step


class MultiModalTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infonce_loss = InfoNCELoss().cuda()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(device)
        user_embeddings = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            history_field_embeddings=inputs['history_field_embeddings'],
            history_attention_mask=inputs['history_attention_mask']
        )
        loss = self.infonce_loss(
            user_embeddings,
            inputs['positive_item_embeddings'],
            inputs['negative_item_embeddings'],
            inputs.get('negative_masks', None)
        )
        return (loss, user_embeddings) if return_outputs else loss


def load_embedding_dict(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_amazon_lranker_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_item_dict(item_dict_path):
    """Load item dictionary, ensuring item_id is present in each item's data."""
    with open(item_dict_path, 'r') as f:
        item_dict = json.load(f)
    # The item_dict maps item_id -> item_metadata. The key is the item_id.
    # We need to ensure that the item_id is also present *inside* the metadata dict
    # for consistent access in the Dataset classes.
    for item_id, item_data in item_dict.items():
        if isinstance(item_data, dict) and 'item_id' not in item_data:
            item_data['item_id'] = item_id
    return item_dict


# Base class for joint training datasets to handle Q-Former inputs
class AmazonBeautyJointDataset(Dataset):
    def __init__(self, data, item_emb_dict, tokenizer, item_dict, item_encoder, field_cache_dir, max_length=512, item_emb_dim=1024):
        self.data = data
        self.item_emb_dict = item_emb_dict
        self.tokenizer = tokenizer
        self.item_dict = item_dict
        self.max_length = max_length
        self.item_emb_dim = item_emb_dim
        self.num_history_items = 10
        self.num_query_tokens_per_item = 2
        
        # Initialize QFormerDataset to handle field embedding caching and retrieval
        item_samples = list(self.item_dict.values())
        self.qformer_field_dataset = QFormerDataset(
            samples=item_samples,
            item_encoder=item_encoder,
            cache_dir=field_cache_dir,
            precompute_batch_size=8192
        )
        self.item_id_to_idx = {sample['item_id']: i for i, sample in enumerate(item_samples)}
        
        # Pre-compute zero tensors for padding
        num_fields = len(self.qformer_field_dataset.available_fields)
        # Correctly get field_embedding_dim from the item_encoder
        field_emb_dim = item_encoder.embedding_dim 
        self.zero_field_embeddings = torch.zeros((num_fields, field_emb_dim), dtype=torch.float32)
        self.zero_attention_mask = torch.zeros(num_fields, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def _get_history_qformer_inputs(self, history_item_ids):
        history_field_embeddings = []
        history_attention_mask = []
        
        for i in range(self.num_history_items):
            if i < len(history_item_ids):
                item_id = str(history_item_ids[i]) # Ensure item ID is a string for lookup
                if item_id in self.item_id_to_idx:
                    qformer_data = self.qformer_field_dataset[self.item_id_to_idx[item_id]]
                    history_field_embeddings.append(qformer_data['field_embeddings'])
                    history_attention_mask.append(qformer_data['attention_mask'])
                else:
                    # Item not in dict, use padding
                    history_field_embeddings.append(self.zero_field_embeddings)
                    history_attention_mask.append(self.zero_attention_mask)
            else:
                # Padded history slot
                history_field_embeddings.append(self.zero_field_embeddings)
                history_attention_mask.append(self.zero_attention_mask)
                
        return torch.stack(history_field_embeddings), torch.stack(history_attention_mask)

    def _construct_input_text(self, history):
        history_parts = []
        for i in range(self.num_history_items):
            query_token_part = "".join([f" <|history_item_{i}_query_{j}|>" for j in range(self.num_query_tokens_per_item)])
            if i < len(history):
                item_id = history[i]
                title = self.item_dict.get(item_id, {}).get('title', f"Item {item_id}")
                if len(title) > 80:
                    title = title[:77] + "..."
                history_parts.append(f"{i+1}. {title}{query_token_part}")
            else:
                history_parts.append(query_token_part.strip())
        history_str = ", ".join(history_parts)
        return f"I have bought these items in the past: {history_str}"

# Updated dataset logic for JOINT training
class AmazonBeautyTrainDataset(AmazonBeautyJointDataset):
    def __init__(self, data, item_emb_dict, tokenizer, item_dict, item_encoder, field_cache_dir, max_length=512, max_negatives=10, item_emb_dim=1024):
        super().__init__(data, item_emb_dict, tokenizer, item_dict, item_encoder, field_cache_dir, max_length, item_emb_dim)
        self.max_negatives = max_negatives

    def __getitem__(self, idx):
        item = self.data[idx]
        history = item['history']
        candidates = item['candidate']
        ground_truth = item['ground_truth']

        history_field_embeddings, history_attention_mask = self._get_history_qformer_inputs(history)

        # Candidate item embeddings
        candidate_embeddings = []
        default_item_emb = np.zeros(self.item_emb_dim, dtype=np.float32)
        for item_id in candidates:
            emb = self.item_emb_dict.get(str(item_id), default_item_emb) # Ensure item_id is string
            candidate_embeddings.append(emb)
        candidate_embeddings = np.array(candidate_embeddings, dtype=np.float32)

        gt_index = candidates.index(ground_truth)
        positive_item_embedding = candidate_embeddings[gt_index]
        negative_item_embeddings = np.delete(candidate_embeddings, gt_index, axis=0)

        input_text = self._construct_input_text(history)
        
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'history_field_embeddings': history_field_embeddings,
            'history_attention_mask': history_attention_mask,
            'positive_item_embedding': torch.tensor(positive_item_embedding, dtype=torch.float32),
            'negative_item_embeddings': torch.tensor(negative_item_embeddings, dtype=torch.float32),
        }

class AmazonBeautyValDataset(AmazonBeautyJointDataset):
    def __init__(self, data, item_emb_dict, tokenizer, item_dict, item_encoder, field_cache_dir, max_length=512, item_emb_dim=1024):
        super().__init__(data, item_emb_dict, tokenizer, item_dict, item_encoder, field_cache_dir, max_length, item_emb_dim)

    def __getitem__(self, idx):
        item = self.data[idx]
        history = item['history']
        candidates = item['candidate']
        ground_truth = item['ground_truth']

        history_field_embeddings, history_attention_mask = self._get_history_qformer_inputs(history)

        # Candidate item embeddings
        candidate_embeddings = []
        default_item_emb = np.zeros(self.item_emb_dim, dtype=np.float32)
        for item_id in candidates:
            emb = self.item_emb_dict.get(str(item_id), default_item_emb) # Ensure item_id is string
            candidate_embeddings.append(emb)
        candidate_embeddings = np.array(candidate_embeddings, dtype=np.float32)

        gt_index = candidates.index(ground_truth)
        gt_embedding = candidate_embeddings[gt_index]
        negative_embeddings = [emb for i, emb in enumerate(candidate_embeddings) if i != gt_index]
        positive_item_embedding = np.array(gt_embedding, dtype=np.float32)
        negative_item_embeddings = np.array(negative_embeddings, dtype=np.float32)

        input_text = self._construct_input_text(history)
        
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'history_field_embeddings': history_field_embeddings,
            'history_attention_mask': history_attention_mask,
            'positive_item_embedding': torch.tensor(positive_item_embedding, dtype=torch.float32),
            'negative_item_embeddings': torch.tensor(negative_item_embeddings, dtype=torch.float32),
        }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用！请检查GPU和CUDA安装。")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    base_model_name = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim = 1024
    max_length = 512
    SAVE_STRATEGY = "both"
    USE_WANDB = True

    # Load embedding dicts and item data
    print("Loading item embedding dict...")
    item_emb_dict = load_embedding_dict(ITEM_EMB_PATH)
    print("Loading item dictionary for product titles and Q-Former...")
    item_dict = load_item_dict(ITEM_DICT_PATH) # This dict contains all item metadata
    print("Loading train data...")
    train_data = load_amazon_lranker_data(TRAIN_DATA_PATH)
    print("Loading validation data...")
    val_data = load_amazon_lranker_data(VAL_DATA_PATH)
    
    # Load ItemEncoder and determine field order for Q-Former
    item_encoder = ItemEncoder()
    field_order = get_field_order(item_dict)
    num_fields = len(field_order)
    print(f"Detected {num_fields} unique fields for Q-Former.")
    
    # Load PRE-TRAINED Q-Former, but keep it in train mode for joint fine-tuning
    qformer_model = load_qformer_model(QFORMER_CHECKPOINT_PATH, num_fields=num_fields, eval_mode=False)

    print(f"📁 使用保存策略: {SAVE_STRATEGY}")
    print(f"🔄 JOINTLY training Q-Former and LoRA (no projection)")
    if SAVE_STRATEGY == "best_only":
        print("   - 只有当MRR提升时才会保存模型")
    elif SAVE_STRATEGY == "always":
        print("   - 每次评估都会保存最新模型")
    elif SAVE_STRATEGY == "both":
        print("   - 每次评估都保存最新模型，MRR提升时额外保存最佳模型")

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
    )

    # In main(), update model initialization to include the qformer_model
    model = MultiModalQwenEmbedding(
        base_model_name,
        qformer_model=qformer_model,
        use_lora=True,
        lora_config=lora_config
    ).to(device)

    print("🔍 Model parameter statistics (includes LoRA and Q-Former):")
    model.get_trainable_parameters()

    train_dataset = AmazonBeautyTrainDataset(train_data, item_emb_dict, model.tokenizer, item_dict, item_encoder, FIELD_EMBEDDING_CACHE_DIR, max_length, max_negatives=10, item_emb_dim=embedding_dim)
    val_dataset = AmazonBeautyValDataset(val_data, item_emb_dict, model.tokenizer, item_dict, item_encoder, FIELD_EMBEDDING_CACHE_DIR, max_length, item_emb_dim=embedding_dim)
    
    mrr_evaluator = MRREvaluator(model, model.tokenizer, val_dataset)
    best_mrr_callback = BestMRRCallback(
        mrr_evaluator,
        save_dir="./lora_qformer_model_output_joint_training",
        eval_steps=20,
        save_strategy=SAVE_STRATEGY,
    )
    data_collator = MultiModalDataCollator(model.tokenizer, max_negatives=10)
    training_args = TrainingArguments(
        output_dir="./lora_qformer_embedding_model_output_joint_training",
        per_device_train_batch_size=16,  # Further reduced for joint training memory
        gradient_accumulation_steps=1,   # Added to maintain effective batch size while using more VRAM
        num_train_epochs=500,
        learning_rate=1e-4,
        logging_steps=10,
        fp16=True,  # Enable fp16 for faster training and better memory usage
        bf16=False,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        save_strategy="no",
        eval_strategy="no",
        warmup_steps=20,  # Increased warmup for stability with larger batch size
        max_grad_norm=1.0,
        dataloader_num_workers=8,  # Reduced slightly to avoid CPU bottleneck
        dataloader_prefetch_factor=4,  # Added prefetching for better data loading
        group_by_length=True,  # Group similar length sequences for efficiency
    )
    trainer = MultiModalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[best_mrr_callback],
    )
    print(f"🚀 开始多模态LoRA与Q-Former联合训练...")
    print(f"📁 保存策略: {SAVE_STRATEGY}")
    print(f"📊 训练数据来源: {TRAIN_DATA_PATH}")
    print(f"📊 验证数据来源: {VAL_DATA_PATH}")
    print(f"🔧 使用Q-Former动态处理history_items embedding: History Items ({model.num_query_tokens_per_item} tokens per item)")
    print(f"📝 使用动态构建的问题描述: 基于历史购买记录生成个性化推荐问题")

    print("📊 Initial MRR evaluation on validation set...")
    initial_mrr = mrr_evaluator.evaluate_mrr()
    print(f"📊 Initial MRR Score: {initial_mrr:.4f}")

    trainer.train()
    print("✅ Q-Former与LoRA联合多模态训练完成!")

    print("📊 Final MRR evaluation on validation set...")
    final_mrr = mrr_evaluator.evaluate_mrr()
    print(f"📊 Final MRR Score: {final_mrr:.4f}")
    print(f"📈 MRR Improvement: {final_mrr - initial_mrr:.4f}")

    if SAVE_STRATEGY == "best_only":
        print(f"📁 最佳联合训练LoRA+Q-Former模型已保存到 ./lora_qformer_model_output_joint_training")
        print(f"🏆 最佳MRR分数: {best_mrr_callback.best_mrr:.4f}")
    elif SAVE_STRATEGY == "always":
        print(f"📁 最新联合训练LoRA+Q-Former模型已保存到 ./lora_qformer_model_output_joint_training")
        print(f"🏆 训练过程中最佳MRR分数: {best_mrr_callback.best_mrr:.4f}")
    elif SAVE_STRATEGY == "both":
        print(f"📁 最新联合训练LoRA+Q-Former模型已保存到 ./lora_qformer_model_output_joint_training/latest_model")
        print(f"📁 最佳联合训练LoRA+Q-Former模型已保存到 ./lora_qformer_model_output_joint_training/best_model")
        print(f"🏆 最佳MRR分数: {best_mrr_callback.best_mrr:.4f}")

    print("\n🎯 Final Q-Former+LoRA multimodal model statistics:")
    model.get_trainable_parameters()

if __name__ == "__main__":
    main() 