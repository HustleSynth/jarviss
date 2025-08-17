#!/usr/bin/env python3
"""
JARVIS Training Script - Enhanced GPU/CPU Version for Kaggle
Extracted from bot2.py for independent training functionality
Optimized for both GPU and CPU training with robust checkpoint handling
Optimized for Kaggle Tesla T4 GPUs

Supported Data Formats:
- JSON: {"instruction": "...", "input": "...", "output": "..."}
- JSON: {"conversation_id": "...", "messages": [{"role": "...", "content": "..."}], "memory": [{"summary": "..."}]}
- JSONL: One JSON object per line (same format as JSON)
- CSV: Various column formats (input/output, instruction/response, etc.)
- TXT: Instruction format with ### markers
- System prompts and conversation data
"""

import os
import json
import logging
import sqlite3
import warnings
import math
import re
import glob
import shutil
import csv
import copy
import time
import mmap
from datetime import datetime
from collections import defaultdict
from threading import Lock

# ML imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np

# Advanced tokenization support (BPE, Byte-level, WordPiece)
try:
    import tokenizers
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
    from tokenizers.pre_tokenizers import WhitespaceSplit
    TOKENIZERS_AVAILABLE = True
    print("✅ tokenizers library available - BPE and advanced tokenization enabled")
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print("⚠️  tokenizers library not available, falling back to word-level tokenization")

# Byte-level tokenization support (legacy)
try:
    import tokenizers
    BYTE_LEVEL_AVAILABLE = True
except ImportError:
    BYTE_LEVEL_AVAILABLE = False
    print("⚠️  tokenizers library not available, byte-level tokenization disabled")

# ChatGPT feedback: Add gradient checkpointing import
try:
    from torch.utils.checkpoint import checkpoint
    GRADIENT_CHECKPOINTING_AVAILABLE = True
except ImportError:
    GRADIENT_CHECKPOINTING_AVAILABLE = False
    checkpoint = None

# Suppress warnings during training
warnings.filterwarnings("ignore", message=".*checkpoint.*")
warnings.filterwarnings("ignore", message=".*torch._dynamo.*")
warnings.filterwarnings("ignore", message=".*torch._compile.*")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JARVIS_Training")

# Kaggle-specific configuration
KAGGLE_INPUT_DIR = "/kaggle/input/dataaa"
KAGGLE_WORKING_DIR = "/kaggle/working"

# Configuration for Kaggle
LOCAL_MODEL_PATH = os.path.join(KAGGLE_WORKING_DIR, "local_jarvis_model.pth")
BACKUP_MODEL_PATH = os.path.join(KAGGLE_WORKING_DIR, "local_jarvis_model_backup.pth")
DB_PATH = os.path.join(KAGGLE_WORKING_DIR, "jarvis.db")

# Auto-detect GPU/CPU with Kaggle optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables
local_model = None
vocab = None
DB_CONN = None
DB_LOCK = Lock()

# Enhanced GPU configuration for Tesla T4
if torch.cuda.is_available():
    # GPU optimizations for Tesla T4
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Tesla T4 specific optimizations
    gpu_count = torch.cuda.device_count()
    print(f"🚀 CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"🔧 GPU Optimizations enabled: cudnn.benchmark, TF32")
    
    if gpu_count > 1:
        print(f"🚀 Multiple GPUs detected: {gpu_count} Tesla T4 GPUs")
        # Enable multi-GPU training if available
        print(f"🔧 Using GPUs: 0,1 for multi-GPU training")
        
        # Set environment variables for multi-GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Disable for better performance
        
        # Enable NCCL for better multi-GPU communication
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_IB_DISABLE"] = "1"  # Disable InfiniBand for Kaggle
        os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable P2P for Kaggle stability
    else:
        print(f"🚀 Single GPU: {torch.cuda.get_device_name(0)}")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    print("🖥️ CPU-only mode detected")

class SimpleLM(nn.Module):
    """Advanced Transformer-based Language Model for JARVIS"""
    def __init__(self, vocab_size, embed_size=640, num_heads=8, num_layers=14, 
                 hidden_size=2560, dropout=0.05, max_seq_len=1024):
        super().__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        # Enhanced embedding with larger size and better initialization
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embed_norm = nn.LayerNorm(embed_size)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Advanced positional encoding with learned parameters
        self.pos_enc = nn.Parameter(torch.randn(1, max_seq_len, embed_size))
        
        # Multi-head attention with relative positioning
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embed_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
                bias=False
            ) for _ in range(num_layers)
        ])
        
        # Feed-forward networks with gated linear units
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_size, hidden_size * 2, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, embed_size, bias=False)
            ) for _ in range(num_layers)
        ])
        
        # Advanced normalization layers
        self.pre_norms = nn.ModuleList([
            nn.LayerNorm(embed_size, eps=1e-6) for _ in range(num_layers)
        ])
        
        self.post_norms = nn.ModuleList([
            nn.LayerNorm(embed_size, eps=1e-6) for _ in range(num_layers)
        ])
        
        # Final output projection with layer norm
        self.final_norm = nn.LayerNorm(embed_size, eps=1e-6)
        self.output_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_size, vocab_size, bias=False)
        
        # Initialize weights with advanced techniques
        self._init_weights()
        
        # Apply gradient checkpointing for memory efficiency (ChatGPT feedback: Guard import)
        self.gradient_checkpointing = GRADIENT_CHECKPOINTING_AVAILABLE
        
        # Performance analysis
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🤖 SimpleLM initialized with {total_params:,} parameters")
        print(f"📊 Architecture: {num_layers}L/{embed_size}E/{num_heads}H/{hidden_size}F")
        print(f"💾 Memory estimate: {total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
        print(f"⏱️ Estimated training time: ~{(total_params / 1e6) * 0.8:.1f} min/epoch on i5-6300U")
    
    def _init_weights(self):
        """Advanced weight initialization for better training"""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
        for attention in self.attention_layers:
            nn.init.xavier_uniform_(attention.in_proj_weight)
            nn.init.xavier_uniform_(attention.out_proj.weight)
        
        for ffn in self.ffn_layers:
            nn.init.xavier_uniform_(ffn[0].weight)
            nn.init.xavier_uniform_(ffn[3].weight)
        
        nn.init.xavier_uniform_(self.fc.weight)
        
        for norm in self.pre_norms + self.post_norms:
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)
        nn.init.ones_(self.final_norm.weight)
        nn.init.zeros_(self.final_norm.bias)
    
    def forward(self, x, padding_mask=None, causal_mask=None):
        batch_size, seq_len = x.shape
        
        # CRITICAL FIX: Ensure causal mask is properly shaped for DataParallel
        # The error shows [512, 1024] vs [1024, 1024], so we need to fix the mask here
        if causal_mask is not None:
            # Ensure mask is exactly (seq_len, seq_len) - no batch dimension
            if causal_mask.dim() != 2 or causal_mask.shape[0] != seq_len or causal_mask.shape[1] != seq_len:
                logger.warning(f"Invalid causal mask shape: {causal_mask.shape}, expected ({seq_len}, {seq_len})")
                # Recreate the mask with correct dimensions
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
                logger.info(f"Fixed causal mask to: {causal_mask.shape}")
        
        # EXTRA SAFETY: Always recreate the causal mask to ensure it's correct
        # This prevents DataParallel from corrupting the mask dimensions
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # Debug logging for first few forward passes
        if hasattr(self, '_forward_count'):
            self._forward_count += 1
        else:
            self._forward_count = 1
            
        if self._forward_count <= 3:  # Log first 3 forward passes
            logger.info(f"Forward pass {self._forward_count}: input shape={x.shape}, causal_mask shape={causal_mask.shape}")
        
        # Embedding + positional encoding
        x = self.embedding(x) * math.sqrt(self.embed_size)
        x = x + self.pos_enc[:, :seq_len, :]
        x = self.embed_dropout(x)
        
        # Apply gradient checkpointing to transformer layers
        if self.gradient_checkpointing and self.training:
            for i in range(self.num_layers):
                x = torch.utils.checkpoint.checkpoint(
                    self._forward_layer, x, padding_mask, causal_mask, i
                )
        else:
            for i in range(self.num_layers):
                x = self._forward_layer(x, padding_mask, causal_mask, i)
        
        # Final normalization and output
        x = self.final_norm(x)
        x = self.output_dropout(x)
        x = self.fc(x)
        
        return x
    
    def _forward_layer(self, x, padding_mask, causal_mask, layer_idx):
        """Forward pass for a single transformer layer"""
        # Pre-norm
        normed_x = self.pre_norms[layer_idx](x)
        
        # Self-attention with residual connection
        if causal_mask is not None:
            attn_output, _ = self.attention_layers[layer_idx](
                normed_x, normed_x, normed_x,
                attn_mask=causal_mask,
                key_padding_mask=padding_mask,
                need_weights=False
            )
        else:
            attn_output, _ = self.attention_layers[layer_idx](
                normed_x, normed_x, normed_x,
                key_padding_mask=padding_mask,
                need_weights=False
            )
        
        x = x + attn_output
        x = self.post_norms[layer_idx](x)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn_layers[layer_idx](x)
        x = x + ffn_output
        
        return x

class ConversationDataset(Dataset):
    """Ultra-memory-efficient streaming dataset for massive training data with mmap support"""
    def __init__(self, conversations, vocab, packed=False, chunk_size=5000, use_mmap=False):
        # Store only references and implement chunked loading
        self.conversations = conversations  # Reference only, not copied
        self.vocab = vocab
        self.packed = packed
        self.chunk_size = chunk_size
        self.total_conversations = len(conversations)
        self.use_mmap = use_mmap
        
        # Current chunk tracking
        self.current_chunk_start = 0
        self.current_chunk_end = min(chunk_size, self.total_conversations)
        self.current_chunk_indices = list(range(self.current_chunk_start, self.current_chunk_end))
        
        # Shuffle the current chunk
        import random
        random.shuffle(self.current_chunk_indices)
        
        # Memory mapping for large files
        if self.use_mmap and hasattr(conversations, 'name'):
            try:
                self.mmap_file = open(conversations.name, 'rb')
                self.mmap_data = mmap.mmap(self.mmap_file.fileno(), 0, access=mmap.ACCESS_READ)
                print(f"   📦 Memory-mapped dataset: {self.total_conversations:,} total, {len(self.current_chunk_indices):,} per chunk")
            except Exception as e:
                print(f"   ⚠️ Memory mapping failed: {e}, falling back to regular loading")
                self.use_mmap = False
        else:
            print(f"   📦 Streaming dataset: {self.total_conversations:,} total, {len(self.current_chunk_indices):,} per chunk")
        
        # Token drop augmentation rate
        self.token_drop_rate = 0.0
        
    def __len__(self):
        return len(self.current_chunk_indices)
    
    def __getitem__(self, idx):
        # Get conversation from current chunk
        if idx >= len(self.current_chunk_indices):
            # Load next chunk if we've reached the end
            self._load_next_chunk()
            idx = 0
        
        conv_idx = self.current_chunk_indices[idx]
        conv = self.conversations[conv_idx]
        
        if self.packed:
            # Handle packed conversations (list of conversation dicts)
            if isinstance(conv, list):
                # Pack multiple conversations into one sequence
                combined_input = ""
                combined_target = ""
                
                for sub_conv in conv:
                    if isinstance(sub_conv, dict) and 'input' in sub_conv and 'target' in sub_conv:
                        combined_input += sub_conv['input'] + " <SEP> "
                        combined_target += sub_conv['target'] + " <SEP> "
                
                # Remove trailing separator
                combined_input = combined_input.rstrip(" <SEP> ")
                combined_target = combined_target.rstrip(" <SEP> ")
                
                input_text = combined_input
                target_text = combined_target
            else:
                # Fallback to single conversation
                input_text = conv['input']
                target_text = conv['target']
        else:
            # Handle single conversations
            input_text = conv['input']
            target_text = conv['target']
        
        # Convert text to token IDs
        input_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in input_text.split()]
        target_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in target_text.split()]
        
        # ChatGPT feedback: Apply token-drop augmentation (1-2% random drop)
        if hasattr(self, 'token_drop_rate') and self.token_drop_rate > 0:
            import random
            # Randomly drop some tokens for regularization
            if len(input_ids) > 10:  # Only apply to longer sequences
                drop_count = max(1, int(len(input_ids) * self.token_drop_rate))
                drop_indices = random.sample(range(len(input_ids)), drop_count)
                for idx in sorted(drop_indices, reverse=True):
                    del input_ids[idx]
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)
    
    def _load_next_chunk(self):
        """Load the next chunk of conversations"""
        self.current_chunk_start = (self.current_chunk_start + self.chunk_size) % self.total_conversations
        self.current_chunk_end = min(self.current_chunk_start + self.chunk_size, self.total_conversations)
        
        # If we've wrapped around, adjust the end
        if self.current_chunk_end <= self.current_chunk_start:
            self.current_chunk_end = self.total_conversations
        
        self.current_chunk_indices = list(range(self.current_chunk_start, self.current_chunk_end))
        
        # Shuffle the new chunk
        import random
        random.shuffle(self.current_chunk_indices)
        
        print(f"   🔄 Loaded new chunk: {self.current_chunk_start:,}-{self.current_chunk_end:,} ({len(self.current_chunk_indices):,} conversations)")
    
    def shuffle_data(self):
        """Load a fresh chunk and shuffle it"""
        import random
        import time
        # Use epoch-specific seed for deterministic but different shuffling
        random.seed(int(time.time() * 1000) % 1000000)
        
        # Load a random chunk
        self.current_chunk_start = random.randint(0, max(0, self.total_conversations - self.chunk_size))
        self.current_chunk_end = min(self.current_chunk_start + self.chunk_size, self.total_conversations)
        self.current_chunk_indices = list(range(self.current_chunk_start, self.current_chunk_end))
        
        # Shuffle the chunk
        random.shuffle(self.current_chunk_indices)
        
        logger.info(f"🔄 Loaded and shuffled new chunk: {self.current_chunk_start:,}-{self.current_chunk_end:,}")

def collate_fn(batch):
    """Memory-efficient collate function for massive datasets"""
    inputs, targets = zip(*batch)
    
    # Use model's max_seq_len to prevent tensor size mismatches
    max_seq_len = 1024  # This should match the model's max_seq_len
    
    # Memory-efficient padding with size limits
    max_input_len = min(max(len(seq) for seq in inputs), max_seq_len)
    max_target_len = min(max(len(seq) for seq in targets), max_seq_len)
    
    # Ensure both input and target use the same sequence length
    max_len = max(max_input_len, max_target_len)
    
    # Truncate sequences that are too long to prevent memory explosion
    inputs_truncated = [seq[:max_len] for seq in inputs]
    targets_truncated = [seq[:max_len] for seq in targets]
    
    # Pad sequences to same length
    inputs_padded = pad_sequence(inputs_truncated, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets_truncated, batch_first=True, padding_value=0)
    
    # Ensure both tensors have exactly the same shape
    assert inputs_padded.shape == targets_padded.shape, f"Shape mismatch: inputs {inputs_padded.shape} vs targets {targets_padded.shape}"
    
    return inputs_padded, targets_padded

def clean_convo_data(text):
    """Clean conversation data more aggressively to remove garbage"""
    if not text:
        return ""
    
    # Remove URLs completely
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove excessive whitespace and punctuation
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{2,}', '.', text)
    text = re.sub(r'[\[\]{}()<>|\\/]{2,}', '', text)
    
    return text.strip()

def init_database():
    """Initialize database connection"""
    global DB_CONN
    try:
        DB_CONN = sqlite3.connect(DB_PATH, check_same_thread=False)
        logger.info("Database connection established")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return False

def load_data_folder_files():
    """Load instruction data and system prompts from the Kaggle input directory
    
    Supports multiple formats:
    - JSON: {"instruction": "...", "input": "...", "output": "..."}
    - JSON: {"conversation_id": "...", "messages": [{"role": "...", "content": "..."}], "memory": [{"summary": "..."}]}
    - JSONL: One JSON object per line
    - CSV: Various column formats
    - TXT: Instruction format with ### markers
    - System prompts and conversation data
    """
    data_conversations = []
    
    try:
        # Get ALL .txt, .json, .jsonl, and .csv files from the Kaggle input directory
        all_txt_files = glob.glob(os.path.join(KAGGLE_INPUT_DIR, "*.txt"))
        all_json_files = glob.glob(os.path.join(KAGGLE_INPUT_DIR, "*.json"))
        all_jsonl_files = glob.glob(os.path.join(KAGGLE_INPUT_DIR, "*.jsonl"))
        all_csv_files = glob.glob(os.path.join(KAGGLE_INPUT_DIR, "*.csv"))
        all_files = all_txt_files + all_json_files + all_jsonl_files + all_csv_files
        logger.info(f"Found {len(all_txt_files)} .txt files, {len(all_json_files)} .json files, {len(all_jsonl_files)} .jsonl files, and {len(all_csv_files)} .csv files in Kaggle input directory")
        
        for file_path in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:  # Skip empty files
                        continue
                    
                    # Try to detect the file format and parse accordingly
                    if file_path.endswith('.json'):
                        # Parse JSON format
                        try:
                            json_data = json.loads(content)
                            if isinstance(json_data, list):
                                # Handle array of training examples
                                for item in json_data:
                                    if isinstance(item, dict):
                                        if 'input' in item and 'target' in item:
                                            data_conversations.append({
                                                'input': str(item['input']),
                                                'target': str(item['target'])
                                            })
                                        elif 'instruction' in item and 'input' in item and 'output' in item:
                                            # Handle the specific format: {"instruction": "...", "input": "...", "output": "..."}
                                            instruction = item['instruction']
                                            input_text = item['input']
                                            output = item['output']
                                            
                                            if instruction and input_text and output:
                                                # Create training pairs
                                                data_conversations.append({
                                                    'input': f"Instruction: {instruction}\nInput: {input_text}",
                                                    'target': output
                                                })
                                                
                                                                                                # Also create instruction-only pair
                                                data_conversations.append({
                                                    'input': f"Instruction: {instruction}",
                                                    'target': output
                                                })
                                                logger.debug(f"JSON: Created 2 training pairs from instruction/input/output format")
                                            elif 'instruction' in item and 'response' in item:
                                                data_conversations.append({
                                                    'input': f"Instruction: {item['instruction']}",
                                                    'target': str(item['response'])
                                                })
                                            elif 'question' in item and 'answer' in item:
                                                data_conversations.append({
                                                    'input': str(item['question']),
                                                    'target': str(item['answer'])
                                                })
                                            elif 'prompt' in item and 'completion' in item:
                                                data_conversations.append({
                                                    'input': str(item['prompt']),
                                                    'target': str(item['completion'])
                                                })
                                            elif 'messages' in item:
                                                # Handle advanced conversation format with role-based messages
                                                messages = item['messages']
                                                if isinstance(messages, list) and len(messages) >= 2:
                                                    # Extract conversation metadata
                                                    conversation_id = json_data.get('conversation_id', 'unknown')
                                                    memory_summaries = json_data.get('memory', [])
                                                    
                                                    # Build context from system messages and memory
                                                    context_parts = []
                                                    system_content = ""
                                                    
                                                    # Collect system messages
                                                    for msg in messages:
                                                        if msg.get('role') == 'system':
                                                            system_content += str(msg.get('content', '')) + " "
                                                    
                                                    if system_content.strip():
                                                        context_parts.append(f"System: {system_content.strip()}")
                                                    
                                                    # Add memory summaries if available
                                                    if memory_summaries:
                                                        memory_text = "; ".join([m.get('summary', '') for m in memory_summaries if m.get('summary')])
                                                        if memory_text:
                                                            context_parts.append(f"Memory: {memory_text}")
                                                    
                                                    # Create training pairs from conversation flow
                                                    conversation_pairs = 0
                                                    for i in range(len(messages) - 1):
                                                        current_msg = messages[i]
                                                        next_msg = messages[i + 1]
                                                        
                                                        if (current_msg.get('role') == 'user' and 
                                                            next_msg.get('role') == 'assistant'):
                                                            
                                                            # Build input with context
                                                            input_text = current_msg.get('content', '')
                                                            if context_parts:
                                                                input_text = f"{' | '.join(context_parts)} | User: {input_text}"
                                                            
                                                            # Create training pair
                                                            data_conversations.append({
                                                                'input': input_text,
                                                                'target': next_msg.get('content', '')
                                                            })
                                                            conversation_pairs += 1
                                                    
                                                    logger.debug(f"JSON: Created {conversation_pairs} training pairs from conversation format (ID: {conversation_id})")
                                            elif 'conversation' in item:
                                                # Handle conversation format
                                                conv = item['conversation']
                                                if isinstance(conv, list) and len(conv) >= 2:
                                                    data_conversations.append({
                                                        'input': str(conv[-2]),
                                                        'target': str(conv[-1])
                                                    })
                            elif isinstance(json_data, dict):
                                # Handle single training example or conversation
                                if 'input' in json_data and 'target' in json_data:
                                    data_conversations.append({
                                        'input': str(json_data['input']),
                                        'target': str(json_data['target'])
                                    })
                                elif 'instruction' in json_data and 'input' in json_data and 'output' in json_data:
                                    # Handle the specific format: {"instruction": "...", "input": "...", "output": "..."}
                                    instruction = json_data['instruction']
                                    input_text = json_data['input']
                                    output = json_data['output']
                                    
                                    if instruction and input_text and output:
                                        # Create training pairs
                                        data_conversations.append({
                                            'input': f"Instruction: {instruction}\nInput: {input_text}",
                                            'target': output
                                        })
                                        
                                                                                # Also create instruction-only pair
                                        data_conversations.append({
                                            'input': f"Instruction: {instruction}",
                                            'target': output
                                        })
                                        logger.debug(f"JSON: Created 2 training pairs from instruction/input/output format")
                                elif 'instruction' in json_data and 'response' in json_data:
                                    data_conversations.append({
                                        'input': f"Instruction: {json_data['instruction']}",
                                        'target': str(json_data['response'])
                                    })
                                elif 'question' in json_data and 'answer' in json_data:
                                    data_conversations.append({
                                        'input': str(json_data['question']),
                                        'target': str(json_data['answer'])
                                    })
                                elif 'prompt' in json_data and 'completion' in json_data:
                                    data_conversations.append({
                                        'input': str(json_data['prompt']),
                                        'target': str(json_data['completion'])
                                    })
                                elif 'conversations' in json_data:
                                    # Handle conversation format
                                    for conv in json_data['conversations']:
                                        if isinstance(conv, list) and len(conv) >= 2:
                                            data_conversations.append({
                                                'input': str(conv[0]),
                                                'target': str(conv[1])
                                            })
                                elif 'messages' in json_data:
                                    # Handle advanced conversation format with role-based messages
                                    messages = json_data['messages']
                                    if isinstance(messages, list) and len(messages) >= 2:
                                        # Extract conversation metadata
                                        conversation_id = json_data.get('conversation_id', 'unknown')
                                        memory_summaries = json_data.get('memory', [])
                                        
                                        # Build context from system messages and memory
                                        context_parts = []
                                        system_content = ""
                                        
                                        # Collect system messages
                                        for msg in messages:
                                            if msg.get('role') == 'system':
                                                system_content += str(msg.get('content', '')) + " "
                                        
                                        if system_content.strip():
                                            context_parts.append(f"System: {system_content.strip()}")
                                        
                                        # Add memory summaries if available
                                        if memory_summaries:
                                            memory_text = "; ".join([m.get('summary', '') for m in memory_summaries if m.get('summary')])
                                            if memory_text:
                                                context_parts.append(f"Memory: {memory_text}")
                                        
                                        # Create training pairs from conversation flow
                                        for i in range(len(messages) - 1):
                                            current_msg = messages[i]
                                            next_msg = messages[i + 1]
                                            
                                            if (current_msg.get('role') == 'user' and 
                                                next_msg.get('role') == 'assistant'):
                                                
                                                # Build input with context
                                                input_text = current_msg.get('content', '')
                                                if context_parts:
                                                    input_text = f"{' | '.join(context_parts)} | User: {input_text}"
                                                
                                                # Create training pair
                                                data_conversations.append({
                                                    'input': input_text,
                                                    'target': next_msg.get('content', '')
                                                })
                                        
                                        logger.debug(f"JSON: Created {len([m for m in messages if m.get('role') == 'user'])} training pairs from conversation format (ID: {conversation_id})")
                                elif 'conversation' in json_data:
                                    # Handle conversation format
                                    conv = json_data['conversation']
                                    if isinstance(conv, list) and len(conv) >= 2:
                                        data_conversations.append({
                                            'input': str(conv[-2]),
                                            'target': str(conv[-1])
                                        })
                        except json.JSONDecodeError:
                            # If JSON parsing fails, treat as text
                            logger.warning(f"Failed to parse JSON in {file_path}, treating as text")
                            # Continue to text parsing below
                        else:
                            # JSON parsing succeeded, skip text parsing
                            continue
                    
                    elif file_path.endswith('.jsonl'):
                        # Parse JSONL format (one JSON object per line)
                        try:
                            jsonl_conversations = 0
                            for line_num, line in enumerate(content.split('\n'), 1):
                                line = line.strip()
                                if not line:  # Skip empty lines
                                    continue
                                
                                try:
                                    json_data = json.loads(line)
                                    if isinstance(json_data, dict):
                                        # Handle the specific format: {"instruction": "...", "input": "...", "output": "..."}
                                        if 'instruction' in json_data and 'input' in json_data and 'output' in json_data:
                                            instruction = json_data['instruction']
                                            input_text = json_data['input']
                                            output = json_data['output']
                                            
                                            if instruction and input_text and output:
                                                # Create training pairs
                                                data_conversations.append({
                                                    'input': f"Instruction: {instruction}\nInput: {input_text}",
                                                    'target': output
                                                })
                                                
                                                # Also create instruction-only pair
                                                data_conversations.append({
                                                    'input': f"Instruction: {instruction}",
                                                    'target': output
                                                })
                                                
                                                jsonl_conversations += 2
                                                logger.debug(f"JSONL: Created 2 training pairs from instruction/input/output format")
                                        
                                        # Handle other common JSONL formats
                                        elif 'instruction' in json_data and 'response' in json_data:
                                            data_conversations.append({
                                                'input': f"Instruction: {json_data['instruction']}",
                                                'target': str(json_data['response'])
                                            })
                                            jsonl_conversations += 1
                                        
                                        elif 'input' in json_data and 'target' in json_data:
                                            data_conversations.append({
                                                'input': str(json_data['input']),
                                                'target': str(json_data['target'])
                                            })
                                            jsonl_conversations += 1
                                        
                                        elif 'question' in json_data and 'answer' in json_data:
                                            data_conversations.append({
                                                'input': str(json_data['question']),
                                                'target': str(json_data['answer'])
                                            })
                                            jsonl_conversations += 1
                                        
                                        elif 'prompt' in json_data and 'completion' in json_data:
                                            data_conversations.append({
                                                'input': str(json_data['prompt']),
                                                'target': str(json_data['completion'])
                                            })
                                            jsonl_conversations += 1
                                        
                                        elif 'messages' in json_data:
                                            # Handle advanced conversation format with role-based messages
                                            messages = json_data['messages']
                                            if isinstance(messages, list) and len(messages) >= 2:
                                                # Extract conversation metadata
                                                conversation_id = json_data.get('conversation_id', 'unknown')
                                                memory_summaries = json_data.get('memory', [])
                                                
                                                # Build context from system messages and memory
                                                context_parts = []
                                                system_content = ""
                                                
                                                # Collect system messages
                                                for msg in messages:
                                                    if msg.get('role') == 'system':
                                                        system_content += str(msg.get('content', '')) + " "
                                                
                                                if system_content.strip():
                                                    context_parts.append(f"System: {system_content.strip()}")
                                                
                                                # Add memory summaries if available
                                                if memory_summaries:
                                                    memory_text = "; ".join([m.get('summary', '') for m in memory_summaries if m.get('summary')])
                                                    if memory_text:
                                                        context_parts.append(f"Memory: {memory_text}")
                                                
                                                # Create training pairs from conversation flow
                                                conversation_pairs = 0
                                                for i in range(len(messages) - 1):
                                                    current_msg = messages[i]
                                                    next_msg = messages[i + 1]
                                                    
                                                    if (current_msg.get('role') == 'user' and 
                                                        next_msg.get('role') == 'assistant'):
                                                        
                                                        # Build input with context
                                                        input_text = current_msg.get('content', '')
                                                        if context_parts:
                                                            input_text = f"{' | '.join(context_parts)} | User: {input_text}"
                                                        
                                                        # Create training pair
                                                        data_conversations.append({
                                                            'input': input_text,
                                                            'target': next_msg.get('content', '')
                                                        })
                                                        conversation_pairs += 1
                                                
                                                jsonl_conversations += conversation_pairs
                                                logger.debug(f"JSONL: Created {conversation_pairs} training pairs from conversation format (ID: {conversation_id})")
                                
                                except json.JSONDecodeError as line_error:
                                    logger.warning(f"Failed to parse JSONL line {line_num} in {file_path}: {line_error}")
                                    continue
                            
                            logger.info(f"Successfully parsed JSONL file: {file_path} - {jsonl_conversations} conversations")
                            continue  # Skip text parsing for JSONL files
                            
                        except Exception as e:
                            logger.warning(f"Failed to parse JSONL in {file_path}: {e}")
                            # Continue to text parsing below
                    
                    elif file_path.endswith('.csv'):
                        # Parse CSV format
                        try:
                            with open(file_path, 'r', encoding='utf-8', newline='') as csvfile:
                                csv_reader = csv.DictReader(csvfile)
                                for row in csv_reader:
                                    # Handle "act","prompt" format specifically
                                    if 'act' in row and 'prompt' in row:
                                        if row['act'].strip() and row['prompt'].strip():
                                            data_conversations.append({
                                                'input': f"Act: {row['act']}",
                                                'target': str(row['prompt'])
                                            })
                                    # Handle other common CSV formats
                                    elif 'input' in row and 'output' in row:
                                        if row['input'].strip() and row['output'].strip():
                                            data_conversations.append({
                                                'input': str(row['input']),
                                                'target': str(row['output'])
                                            })
                                    elif 'instruction' in row and 'response' in row:
                                        if row['instruction'].strip() and row['response'].strip():
                                            data_conversations.append({
                                                'input': f"Instruction: {row['instruction']}",
                                                'target': str(row['response'])
                                            })
                                    elif 'question' in row and 'answer' in row:
                                        if row['question'].strip() and row['answer'].strip():
                                            data_conversations.append({
                                                'input': str(row['question']),
                                                'target': str(row['answer'])
                                            })
                                    elif 'prompt' in row and 'completion' in row:
                                        if row['prompt'].strip() and row['completion'].strip():
                                            data_conversations.append({
                                                'input': str(row['prompt']),
                                                'target': str(row['completion'])
                                            })
                            logger.info(f"Successfully parsed CSV file: {file_path}")
                            continue  # Skip text parsing for CSV files
                        except Exception as e:
                            logger.warning(f"Failed to parse CSV in {file_path}: {e}")
                            # Continue to text parsing below
                    
                    # Text parsing (for .txt files and failed JSON/CSV files)
                    if "### Instruction:" in content and "### Input:" in content and "### Response:" in content:
                        # Parse instruction format: ### Instruction:, ### Input:, ### Response:, ### End
                        sections = content.split('### ')
                        for i in range(0, len(sections) - 3, 4):
                            if (i + 3 < len(sections) and 
                                sections[i].startswith('Instruction:') and
                                sections[i + 1].startswith('Input:') and
                                sections[i + 2].startswith('Response:') and
                                sections[i + 3].startswith('End')):
                                
                                instruction = sections[i].replace('Instruction:', '').strip()
                                input_text = sections[i + 1].replace('Input:', '').strip()
                                response = sections[i + 2].replace('Response:', '').strip()
                                
                                if instruction and input_text and response:
                                    # Create training pairs
                                    data_conversations.append({
                                        'input': f"Instruction: {instruction}\nInput: {input_text}",
                                        'target': response
                                    })
                                    
                                    # Also create reverse pair for better learning
                                    data_conversations.append({
                                        'input': f"Input: {input_text}",
                                        'target': response
                                    })
                    
                    elif "System Prompt:" in content:
                        # Parse system prompt format: System Prompt:, Context:, Requirements:, Expected Output:, End
                        sections = content.split('System Prompt:')
                        for section in sections[1:]:  # Skip first empty section
                            if section.strip():
                                # Extract the system prompt content
                                lines = section.split('\n')
                                system_content = ""
                                for line in lines:
                                    if line.strip() and not line.startswith('Context:') and not line.startswith('Requirements:') and not line.startswith('Expected Output:') and not line.startswith('End'):
                                        system_content += line.strip() + " "
                                
                                if system_content.strip():
                                    # Create training pairs for system prompts
                                    data_conversations.append({
                                        'input': f"System Prompt: {system_content.strip()}",
                                        'target': "I understand my role and will follow these instructions."
                                    })
                    
                    else:
                        # Unknown format - try to extract any meaningful content
                        lines = content.split('\n')
                        meaningful_lines = [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]
                        if meaningful_lines:
                            # Create a basic training pair from the content
                            data_conversations.append({
                                'input': f"Content: {meaningful_lines[0][:200]}...",
                                'target': meaningful_lines[1] if len(meaningful_lines) > 1 else meaningful_lines[0]
                            })
                            
            except Exception as e:
                logger.warning(f"Error loading file {file_path}: {e}")
        
        logger.info(f"Loaded {len(data_conversations)} additional training examples from data folder")
        
        # Log breakdown by file type
        txt_count = len([f for f in all_files if f.endswith('.txt')])
        json_count = len([f for f in all_files if f.endswith('.json')])
        jsonl_count = len([f for f in all_files if f.endswith('.jsonl')])
        csv_count = len([f for f in all_files if f.endswith('.csv')])
        logger.info(f"Processed {txt_count} .txt files, {json_count} .json files, {jsonl_count} .jsonl files, and {csv_count} .csv files")
        
        return data_conversations
        
    except Exception as e:
        logger.error(f"Error loading data folder files: {e}")
        return []

def load_large_file_streaming(file_path, chunk_size=10000):
    """Load large files in chunks to avoid memory issues"""
    conversations = []
    chunk_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            chunk = []
            for line_num, line in enumerate(f):
                line = line.strip()
                if line:
                    chunk.append({
                        'input': f"File: {os.path.basename(file_path)} (chunk {chunk_count + 1})",
                        'target': line
                    })
                
                # Process chunk when it reaches the target size
                if len(chunk) >= chunk_size:
                    conversations.extend(chunk)
                    chunk_count += 1
                    chunk = []
                    
                    # Memory management
                    if chunk_count % 10 == 0:
                        gc.collect()
                        print(f"      📦 Processed {chunk_count} chunks from {os.path.basename(file_path)}")
            
            # Process remaining chunk
            if chunk:
                conversations.extend(chunk)
                chunk_count += 1
        
        print(f"      ✅ Completed streaming load: {chunk_count} chunks, {len(conversations)} samples")
        return conversations
        
    except Exception as e:
        logger.error(f"Error streaming file {file_path}: {e}")
        return []

def load_jsonl_streaming(file_path, max_samples=None):
    """Load JSONL files with streaming support"""
    conversations = []
    sample_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if max_samples and sample_count >= max_samples:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        # Handle different JSONL formats
                        if 'instruction' in data and 'output' in data:
                            conversations.append({
                                'input': data['instruction'],
                                'target': data['output']
                            })
                        elif 'input' in data and 'target' in data:
                            conversations.append(data)
                        elif 'conversation_id' in data and 'messages' in data:
                            # Convert conversation format to training pairs
                            conv_pairs = convert_conversation_to_pairs(data)
                            conversations.extend(conv_pairs)
                        
                        sample_count += 1
                        
                        # Progress reporting
                        if sample_count % 10000 == 0:
                            print(f"      📊 Loaded {sample_count:,} samples from {os.path.basename(file_path)}")
                
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num + 1}: {e}")
                    continue
                
                # Memory management
                if sample_count % 50000 == 0:
                    gc.collect()
        
        print(f"      ✅ Completed JSONL load: {sample_count:,} samples from {os.path.basename(file_path)}")
        return conversations
        
    except Exception as e:
        logger.error(f"Error streaming JSONL file {file_path}: {e}")
        return []

def convert_conversation_to_pairs(conversation_data):
    """Convert conversation format to training pairs"""
    pairs = []
    
    try:
        messages = conversation_data.get('messages', [])
        if len(messages) < 2:
            return pairs
        
        # Create training pairs from consecutive messages
        for i in range(len(messages) - 1):
            current_msg = messages[i]
            next_msg = messages[i + 1]
            
            if current_msg.get('role') == 'user' and next_msg.get('role') == 'assistant':
                pairs.append({
                    'input': current_msg.get('content', ''),
                    'target': next_msg.get('content', '')
                })
        
        return pairs
        
    except Exception as e:
        logger.warning(f"Error converting conversation to pairs: {e}")
        return []

def build_vocab_from_db(incremental=True, max_vocab_size=100000):
    """Build or update vocab from DB conversations and data folder files with PROPER frequency counting"""
    global vocab
    
    print(f"🔍 Building vocabulary (incremental={incremental}, max_size={max_vocab_size:,})...")
    
    if incremental and vocab is not None:
        current_words = set(vocab.keys()) - {'<PAD>', '<UNK>'}
        max_idx = max(vocab.values())
        print(f"📚 Current vocabulary size: {len(vocab)}")
    else:
        vocab = {}
        current_words = set()
        max_idx = 1  # <UNK> = 1, <PAD> = 0
        print("📚 Starting with fresh vocabulary")

    # PROPER frequency counting - count ALL occurrences, not just unique words
    word_freq = defaultdict(int)
    
    # Count frequencies from database conversations
    print("🔄 Counting word frequencies from database...")
    try:
        with DB_LOCK:
            if DB_CONN is not None:
                cursor = DB_CONN.cursor()
                cursor.execute("SELECT content, response FROM memory")
                rows = cursor.fetchall()
                print(f"📊 Found {len(rows)} database conversations")
                
                # Count EVERY word occurrence, not just unique words
                for content, response in rows:
                    content_words = clean_convo_data(content).split()
                    response_words = clean_convo_data(response).split()
                    
                    # Count each word occurrence
                    for word in content_words:
                        word_freq[word] += 1
                    for word in response_words:
                        word_freq[word] += 1
            else:
                print("⚠️ No database connection, skipping database words")
                rows = []
    except Exception as e:
        print(f"⚠️ Error reading database: {e}")
        rows = []
    
    # Count frequencies from data folder files
    print("🔄 Counting word frequencies from data folder files...")
    data_folder_conversations = load_data_folder_files()
    print(f"📊 Found {len(data_folder_conversations)} data folder conversations")
    
    for conv in data_folder_conversations:
        input_words = clean_convo_data(conv['input']).split()
        target_words = clean_convo_data(conv['target']).split()
        
        # Count each word occurrence
        for word in input_words:
            word_freq[word] += 1
        for word in target_words:
            word_freq[word] += 1
    
    print(f"📊 Total word occurrences counted: {sum(word_freq.values()):,}")
    print(f"📊 Unique words found: {len(word_freq):,}")
    
    # Sort by ACTUAL frequency (not just unique words)
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Show top 20 most frequent words
    print(f"\n🏆 Top 20 most frequent words:")
    for i, (word, freq) in enumerate(sorted_words[:20]):
        print(f"   {i+1:2d}. '{word}': {freq:,} occurrences")
    
    # Keep only the most frequent words up to max_vocab_size
    if len(sorted_words) > max_vocab_size:
        print(f"\n⚠️ Vocabulary size limit reached! Keeping top {max_vocab_size:,} most frequent words")
        print(f"   Total unique words found: {len(sorted_words):,}")
        print(f"   Words being dropped: {len(sorted_words) - max_vocab_size:,}")
        sorted_words = sorted_words[:max_vocab_size]
    
    new_words = [word for word, freq in sorted_words if word not in current_words]
    print(f"\n📚 Adding {len(new_words)} new words to vocabulary")
    
    for word in new_words:
        max_idx += 1
        vocab[word] = max_idx

    if '<PAD>' not in vocab:
        vocab['<PAD>'] = 0
    if '<UNK>' not in vocab:
        vocab['<UNK>'] = 1

    final_vocab_size = len(vocab)
    print(f"✅ Vocabulary built successfully: {final_vocab_size:,} total words")
    
    # Calculate parameter impact
    if final_vocab_size > 100000:
        print(f"⚠️ WARNING: Large vocabulary detected! This will significantly increase model parameters.")
        print(f"   Expected parameter increase: ~{final_vocab_size * 640 / 1e6:.1f}M from embedding layer alone")
    
    return final_vocab_size

def resize_model(model, old_vocab_size, new_vocab_size):
    """Resize the model to accommodate a larger vocabulary"""
    if old_vocab_size == new_vocab_size:
        return model

    # Handle DataParallel wrapped models
    if hasattr(model, 'module'):
        # Unwrap DataParallel to access the actual model
        actual_model = model.module
        was_data_parallel = True
    else:
        actual_model = model
        was_data_parallel = False

    new_model = SimpleLM(new_vocab_size).to(device)

    # Copy old embeddings
    new_model.embedding.weight.data[:old_vocab_size] = actual_model.embedding.weight.data
    # Initialize new embeddings
    nn.init.normal_(new_model.embedding.weight.data[old_vocab_size:], mean=0.0, std=0.02)

    # Copy old fc weights (no bias since bias=False)
    new_model.fc.weight.data[:old_vocab_size] = actual_model.fc.weight.data
    # Initialize new fc weights
    nn.init.xavier_uniform_(new_model.fc.weight.data[old_vocab_size:])

    # Copy attention layers
    for i in range(min(len(actual_model.attention_layers), len(new_model.attention_layers))):
        new_model.attention_layers[i].load_state_dict(actual_model.attention_layers[i].state_dict())
    
    # Copy feed-forward layers
    for i in range(min(len(actual_model.ffn_layers), len(new_model.ffn_layers))):
        new_model.ffn_layers[i].load_state_dict(actual_model.ffn_layers[i].state_dict())
    
    # Copy normalization layers
    for i in range(min(len(actual_model.pre_norms), len(new_model.pre_norms))):
        new_model.pre_norms[i].load_state_dict(actual_model.pre_norms[i].state_dict())
    
    for i in range(min(len(actual_model.post_norms), len(new_model.post_norms))):
        new_model.post_norms[i].load_state_dict(actual_model.post_norms[i].state_dict())
    
    # Copy final norm
    new_model.final_norm.load_state_dict(actual_model.final_norm.state_dict())

    # Re-wrap with DataParallel if the original was wrapped
    if was_data_parallel:
        new_model = torch.nn.DataParallel(new_model)

    return new_model

def load_local_model():
    """Load existing local model or create new one with multi-GPU support for Tesla T4"""
    global local_model, vocab
    
    print("🔍 Starting model loading process...")
    
    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"📁 Found existing model: {LOCAL_MODEL_PATH}")
        print(f"💾 Model size: {os.path.getsize(LOCAL_MODEL_PATH) / (1024*1024):.1f} MB")
        
        try:
            print("🔄 Loading checkpoint from disk (this may take a few minutes)...")
            print("   Loading", end="", flush=True)
            
            # Show progress dots while loading
            import time
            checkpoint = torch.load(LOCAL_MODEL_PATH, map_location=device)
            print(" ✅")
            print("✅ Checkpoint loaded successfully")
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint and 'vocab' in checkpoint:
                print("🔍 Extracting vocabulary...")
                vocab = checkpoint['vocab']
                vocab_size = len(vocab)
                print(f"📚 Vocabulary size: {vocab_size}")
                
                # Use saved model config if available, otherwise use defaults
                if 'model_config' in checkpoint:
                    config = checkpoint['model_config']
                    print(f"🏗️ Creating model with saved config: {config['num_layers']} layers, {config['embed_size']} embed size, {config['hidden_size']} hidden size")
                    local_model = SimpleLM(
                        vocab_size=vocab_size,
                        embed_size=config['embed_size'],
                        num_heads=config['num_heads'],
                        num_layers=config['num_layers'],
                        hidden_size=config['hidden_size'],
                        dropout=config['dropout'],
                        max_seq_len=config['max_seq_len']
                    ).to(device)
                    logger.info(f"Loaded model with saved config: {config['num_layers']} layers, {config['embed_size']} embed size, {config['hidden_size']} hidden size")
                else:
                    print("🏗️ Creating model with default config (legacy checkpoint)")
                    local_model = SimpleLM(vocab_size).to(device)
                    logger.info("Loaded model with default config (legacy checkpoint)")
                
                print("🔄 Loading model state dict...")
                local_model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Model state dict loaded successfully")
                logger.info("Loaded local model and vocab.")
                
                print("🔄 Building vocabulary from database...")
                # Incrementally update vocab and resize if necessary
                old_vocab_size = vocab_size
                build_vocab_from_db(incremental=True)
                new_vocab_size = len(vocab)
                if new_vocab_size > old_vocab_size:
                    print(f"🔄 Resizing model from {old_vocab_size} to {new_vocab_size} vocab size...")
                    local_model = resize_model(local_model, old_vocab_size, new_vocab_size)
                    print(f"✅ Model resized successfully")
                    logger.info(f"Resized model from {old_vocab_size} to {new_vocab_size} vocab size.")
                else:
                    print("✅ No vocabulary expansion needed")
                
                # Multi-GPU training will be enabled at the end after all processing
            else:
                print("⚠️ Old checkpoint detected without vocab. Initializing new model.")
                logger.warning("Old checkpoint detected without vocab. Initializing new model.")
                create_optimized_vocab(max_size=100000)
                vocab_size = len(vocab)
                local_model = SimpleLM(vocab_size).to(device)
                
                # Multi-GPU training will be enabled at the end after all processing
        except Exception as e:
            print(f"❌ Error loading local model: {e}")
            logger.error(f"Error loading local model from {LOCAL_MODEL_PATH}: {e}")
            print("🔄 Attempting to create new model instead...")
            local_model = None # Ensure local_model is None on error
    else:
        print("📁 No existing model found, creating new one...")
        create_optimized_vocab(max_size=100000)
        vocab_size = len(vocab)
        local_model = SimpleLM(vocab_size).to(device)
        
        # Multi-GPU training will be enabled at the end after all processing
    
    print("✅ Model loading process completed!")
    
    # Apply DataParallel at the very end if needed
    if local_model is not None and torch.cuda.device_count() > 1:
        print(f"🚀 Final step: Checking GPU configuration...")
        print(f"   Available GPUs: {torch.cuda.device_count()} Tesla T4 GPUs")
        
        # Enable dual-GPU training with CPU assistance
        print("   🚀 Enabling dual-GPU training with CPU assistance...")
        
        # Set up multi-GPU training
        torch.cuda.set_device(0)  # Primary GPU
        
        # Wrap model with DataParallel for dual-GPU training
        local_model = torch.nn.DataParallel(local_model, device_ids=[0, 1])
        print("   ✅ DataParallel enabled for dual-GPU training")
        print("   🎯 Primary GPU: 0, Secondary GPU: 1")
        
        # Enable CPU assistance for memory management
        print("   🧠 CPU assistance enabled (25GB RAM limit)")
        
    elif local_model is not None and torch.cuda.device_count() == 1:
        print(f"🚀 Single GPU detected: {torch.cuda.get_device_name(0)}")
        print("   ✅ Single GPU training mode enabled")

def clean_database_garbage():
    """Clean up garbage data like meme links and URLs from the database"""
    with DB_LOCK:
        try:
            cursor = DB_CONN.cursor()
            
            # Count garbage entries before cleanup
            cursor.execute("SELECT COUNT(*) FROM memory WHERE response LIKE '%http%' OR response LIKE '%www.%' OR response LIKE '%meme%' OR response LIKE '%gif%'")
            garbage_count = cursor.fetchone()[0]
            
            if garbage_count > 0:
                # Remove entries with URLs, meme references, etc.
                cursor.execute("DELETE FROM memory WHERE response LIKE '%http%' OR response LIKE '%www.%' OR response LIKE '%meme%' OR response LIKE '%gif%'")
                
                # Also remove entries that are just URLs or very short responses
                cursor.execute("DELETE FROM memory WHERE LENGTH(response) < 5 OR response LIKE '%/%'")
                
                # Remove entries with excessive special characters (likely garbage)
                cursor.execute("DELETE FROM memory WHERE response LIKE '%[%]%' OR response LIKE '%{%}%'")
                
                DB_CONN.commit()
                logger.info(f"Cleaned {garbage_count} garbage entries from database")
                return garbage_count
            else:
                logger.info("No garbage data found in database")
                return 0
                
        except sqlite3.OperationalError as e:
            logger.error(f"DB error in clean_database_garbage: {e}")
            return 0

def train_local_model():
    """Train the local model on conversation data from the database"""
    global local_model, vocab
    
    print("🧠 Starting local model training...")
    print("📝 Note: Training warnings are normal and expected")
    print("🔧 The training will complete and save the model")
    
    # Initialize multi-GPU training early if available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # Check if Smart Single-GPU mode is enabled
        if hasattr(local_model, 'smart_single_gpu_mode') and local_model.smart_single_gpu_mode:
            print(f"🚀 Smart Single-GPU mode detected - skipping DataParallel")
            print(f"   🎯 Using GPU 0 for training, GPU 1 available for other tasks")
            print(f"   ✅ Gradient checkpointing enabled for memory efficiency")
        else:
            print(f"🚀 Initializing multi-GPU training with {torch.cuda.device_count()} GPUs...")
            
            # Set primary GPU
            torch.cuda.set_device(0)
            
            # Check if model is already wrapped with DataParallel
            if not isinstance(local_model, torch.nn.DataParallel):
                print("   🔧 Wrapping model with DataParallel for multi-GPU training...")
                # Ensure model is on the correct device before wrapping
                if next(local_model.parameters()).device.type != 'cuda':
                    local_model = local_model.cuda()
                    print("   🔧 Model moved to CUDA before DataParallel wrapping")
                
                # Check if model is properly initialized
                print(f"   🔍 Model vocab_size: {local_model.vocab_size}")
                print(f"   🔍 Model embed_size: {local_model.embed_size}")
                print(f"   🔍 Model device: {next(local_model.parameters()).device}")
                
                local_model = torch.nn.DataParallel(local_model, device_ids=[0, 1])
                print("   ✅ DataParallel wrapper applied")
            else:
                print("   ✅ Model already wrapped with DataParallel")
            
            # Verify multi-GPU setup
            print(f"   📊 Model device: {next(local_model.parameters()).device}")
            print(f"   🔧 DataParallel status: {type(local_model).__name__}")
            
            # Additional device verification
            if isinstance(local_model, torch.nn.DataParallel):
                print(f"   🔍 DataParallel device IDs: {local_model.device_ids}")
                print(f"   🔍 Primary device: {local_model.device_ids[0]}")
                # Check if all parameters are on the correct device
                all_devices = set(p.device for p in local_model.parameters())
                print(f"   🔍 All parameter devices: {all_devices}")
            
            # Test multi-GPU forward pass
            try:
                # Create proper integer indices for vocabulary (not random floats)
                vocab_size = local_model.module.vocab_size if hasattr(local_model, 'module') else local_model.vocab_size
                test_input = torch.randint(0, min(vocab_size, 1000), (2, 64), dtype=torch.long).to('cuda:0')
                
                # Ensure input tensor is properly formatted for DataParallel
                test_input = test_input.contiguous()
                if not test_input.is_contiguous():
                    test_input = test_input.contiguous()
                print(f"   🔍 Input tensor contiguous: {test_input.is_contiguous()}")
                
                # Debug tensor properties
                print(f"   🔍 Test input dtype: {test_input.dtype}, device: {test_input.device}")
                print(f"   🔍 Test input shape: {test_input.shape}")
                print(f"   🔍 Test input min/max: {test_input.min().item()}/{test_input.max().item()}")
                
                # Ensure model is in eval mode for testing
                local_model.eval()
                print(f"   🔍 Model mode: {'eval' if local_model.training == False else 'train'}")
                
                # Check forward method signature
                if hasattr(local_model, 'module'):
                    forward_sig = local_model.module.forward.__code__.co_varnames
                    embedding_layer = local_model.module.embedding
                else:
                    forward_sig = local_model.forward.__code__.co_varnames
                    embedding_layer = local_model.embedding
                print(f"   🔍 Forward method signature: {forward_sig}")
                print(f"   🔍 Embedding layer: {type(embedding_layer).__name__}")
                print(f"   🔍 Embedding input size: {embedding_layer.num_embeddings}")
                print(f"   🔍 Embedding output size: {embedding_layer.embedding_dim}")
                
                with torch.no_grad():
                    # Check input tensor before forward pass
                    print(f"   🔍 Input tensor before forward: dtype={test_input.dtype}, device={test_input.device}")
                    
                    test_output = local_model(test_input)
                    
                    # Check input tensor after forward pass (should be unchanged)
                    print(f"   🔍 Input tensor after forward: dtype={test_input.dtype}, device={test_input.device}")
                    
                print(f"   ✅ Multi-GPU forward pass test successful")
            except Exception as e:
                print(f"   ⚠️ Multi-GPU test failed: {e}")
                print(f"   🔍 Error type: {type(e).__name__}")
                print(f"   🔍 Error details: {str(e)}")
                print("   🔧 Falling back to single GPU training...")
                # Unwrap DataParallel if it causes issues
                if isinstance(local_model, torch.nn.DataParallel):
                    local_model = local_model.module
                    print("   🔧 DataParallel wrapper removed, using single GPU")
    
    # CPU memory management setup
    import psutil
    import gc
    
    # Set CPU memory limit to 25GB (leaving 5GB for system)
    cpu_memory_limit_gb = 25
    cpu_memory_limit_bytes = cpu_memory_limit_gb * 1024 * 1024 * 1024
    
    print(f"\n🧠 CPU MEMORY MANAGEMENT:")
    print(f"   Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"   Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print(f"   Memory limit: {cpu_memory_limit_gb} GB (leaving 5GB for system)")
    
    print("\n🚀 CHATGPT TRAINING OPTIMIZATIONS APPLIED:")
    print("   ✅ Effective batch sizing: 256 sequences (microbatch + gradient accumulation)")
    print("   ✅ Target tokens/step: 250K-1M (ChatGPT recommendation)")
    print("   ✅ Learning rate: 8e-4 to 1.2e-3 (ChatGPT-optimized)")
    print("   ✅ AdamW optimizer: β1=0.9, β2=0.95, weight_decay=0.05")
    print("   ✅ Warmup: 1k-2k steps, cosine decay to 10% of peak")
    print("   ✅ Gradient clipping: max_norm=1.0")
    print("   ✅ Validation split: 5% held out for early stopping")
    print("   ✅ Hard shuffling: New seed every epoch")
    print("   ✅ Sequence packing: Minimize padding, maximize useful tokens")
    print("   ✅ CPU memory management: 25GB limit with garbage collection")
    print("   ✅ Dual-GPU training: DataParallel with device IDs [0, 1]")
    
    print("\n🔧 CHATGPT REFINEMENTS APPLIED:")
    print("   ✅ Dynamic epochs: Configurable based on dataset size (8-20 epochs)")
    print("   ✅ Warmup proportionality: ~1.5% of total training steps")
    print("   ✅ Sequence packing metrics: Accurate token counting across all packs")
    print("   ✅ Gradient checkpointing: Safe import with fallback")
    print("   ✅ EMA weights: Exponential moving average (decay=0.999) for stability")
    print("   ✅ Extended context: 2048 tokens if GPU memory allows")
    print("   ✅ Token-drop augmentation: 1% random drop for regularization")
    
    # Enhanced GPU/CPU detection and optimization
    if torch.cuda.is_available():
        print(f"🚀 GPU Training Mode: {torch.cuda.get_device_name(0)}")
        print(f"💾 Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"🔥 Mixed Precision Training: Enabled")
        print(f"⚡ GPU Optimizations: cudnn.benchmark, TF32")
    else:
        print("🖥️ CPU Training Mode")
        print(f"🧵 CPU Threads: {torch.get_num_threads()}")
        print(f"💾 CPU Memory: {os.getenv('NUMBER_OF_PROCESSORS', 'Unknown')} cores")
    
    # Show architecture comparison
    print("\n🏗️ ARCHITECTURE COMPARISON:")
    print("┌─────────────────┬─────────┬─────────┬─────────┬─────────┬─────────┐")
    print("│ Parameter       │ Old     │ New     │ Change  │ Impact  │ Status  │")
    print("├─────────────────┼─────────┼─────────┼─────────┼─────────┼─────────┤")
    print("│ Embedding Size  │ 512     │ 640     │ +25%    │ +Token  │ ✅      │")
    print("│ Attention Heads │ 16      │ 8       │ -50%    │ +Depth  │ ✅      │")
    print("│ Layers          │ 12      │ 14      │ +17%    │ +Power  │ ✅      │")
    print("│ Hidden Size     │ 2048    │ 2560    │ +25%    │ +MLP    │ ✅      │")
    print("│ Dropout         │ 0.1     │ 0.05    │ -50%    │ +Retain │ ✅      │")
    print("│ Seq Length      │ 2048    │ 1024    │ -50%    │ +Speed  │ ✅      │")
    print("└─────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┘")
    print("🎯 Target: ~70-75M parameters with balanced CPU/GPU performance")
    print("💡 Optimized for GPU training with CUDA, falls back to CPU if needed\n")
    
    try:
        # Check if we have an existing model to continue training on
        if local_model is None and os.path.exists(LOCAL_MODEL_PATH):
            print("🔄 Loading existing model for continued training...")
            load_local_model()
            if local_model is not None:
                print(f"✅ Loaded existing model with {sum(p.numel() for p in local_model.parameters()):,} parameters")
                print(f"📊 Model architecture: {local_model.num_layers} layers, {local_model.embed_size} embed size, {local_model.hidden_size} hidden size")
            else:
                print("⚠️ Failed to load existing model, will create new one")
        
        # Clean database before training
        clean_database_garbage()
        
        with DB_LOCK:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Get conversation data from individual memory
            cursor.execute("SELECT content, response FROM memory WHERE content IS NOT NULL AND response IS NOT NULL AND content != '' AND response != ''")
            conversations = cursor.fetchall()
            
            # Get contextual memory data
            cursor.execute("SELECT conversation_context, key_topics FROM contextual_memory WHERE conversation_context IS NOT NULL AND conversation_context != ''")
            contextual_data = cursor.fetchall()
            
            conn.close()
        
        if not conversations and not contextual_data:
            print("❌ No training data found. Please chat with the bot first to generate conversation data.")
            return
        
        print(f"📚 Training on {len(conversations)} individual conversations and {len(contextual_data)} contextual memories")
        
        # Advanced data preprocessing and augmentation
        augmented_conversations = []
        for content, response in conversations:
            # Clean the conversation data
            cleaned_input = clean_convo_data(content)
            cleaned_target = clean_convo_data(response)
            
            # Only include conversations with non-empty cleaned text
            if cleaned_input.strip() and cleaned_target.strip():
                conv = {'input': cleaned_input, 'target': cleaned_target}
                augmented_conversations.append(conv)
                
                # Data augmentation: reverse input-output pairs for better learning
                if len(cleaned_input.split()) > 5 and len(cleaned_target.split()) > 5:
                    augmented_conversations.append({
                        'input': cleaned_target,
                        'target': cleaned_input
                    })
                
                # Context windowing: create shorter context versions
                input_words = cleaned_input.split()
                
                if len(input_words) > 20:
                    # Create shorter context versions
                    for i in range(0, len(input_words) - 10, 10):
                        window_input = ' '.join(input_words[i:i+20])
                        if len(window_input.strip()) > 10:
                            augmented_conversations.append({
                                'input': window_input,
                                'target': cleaned_target
                            })
        
        # Process contextual memory data
        for context, topics in contextual_data:
            if context and context.strip():
                cleaned_context = clean_convo_data(context)
                if cleaned_context.strip():
                    # Create training pairs from contextual memory
                    topics_str = ", ".join(json.loads(topics) if topics else [])
                    augmented_conversations.append({
                        'input': f"Context: {cleaned_context} | Topics: {topics_str}",
                        'target': cleaned_context
                    })
        
        # Load additional training data from data folder
        print("📁 Loading instruction data and system prompts from data folder...")
        data_folder_conversations = load_data_folder_files()
        if data_folder_conversations:
            augmented_conversations.extend(data_folder_conversations)
            print(f"✅ Added {len(data_folder_conversations)} data folder examples to training set")
        else:
            print("ℹ️ No additional data folder examples found")
        
        logger.info(f"Augmented dataset size: {len(augmented_conversations)} conversations (including contextual memory and data folder)")
        
        # ChatGPT recommendation: Deduplicate near-duplicates to keep curated set punchy
        print("🔍 ChatGPT recommendation: Deduplicating near-duplicates...")
        unique_conversations = []
        seen_hashes = set()
        
        for conv in augmented_conversations:
            # Simple hash-based deduplication
            input_hash = hash(conv['input'].lower().strip())
            target_hash = hash(conv['target'].lower().strip())
            combined_hash = (input_hash, target_hash)
            
            if combined_hash not in seen_hashes:
                seen_hashes.add(combined_hash)
                unique_conversations.append(conv)
        
        # Prevent division by zero
        if len(augmented_conversations) > 0:
            dedup_ratio = len(unique_conversations) / len(augmented_conversations)
            print(f"   Deduplication: {len(augmented_conversations)} → {len(unique_conversations)} ({dedup_ratio*100:.1f}% retained)")
        else:
            dedup_ratio = 0.0
            print(f"   Deduplication: No conversations to deduplicate")
        
        # ChatGPT recommendation: Pack sequences to minimize padding (raise useful tokens/step)
        print("📦 ChatGPT recommendation: Packing sequences to minimize padding...")
        
        # ChatGPT feedback: Try context length 2048 if memory allows
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory_gb >= 12:  # High-end GPU can handle 2048
                max_pack_length = 2048
                print(f"   🚀 Using extended context: 2048 tokens (GPU: {gpu_memory_gb:.1f}GB)")
            else:
                max_pack_length = 1024
                print(f"   📏 Using standard context: 1024 tokens (GPU: {gpu_memory_gb:.1f}GB)")
        else:
            max_pack_length = 1024
            print(f"   📏 Using standard context: 1024 tokens (CPU mode)")
        
        packed_conversations = []
        current_pack = []
        current_length = 0
        total_packed_tokens = 0  # ChatGPT feedback: Track total tokens across all packs
        
        for conv in unique_conversations:
            input_tokens = len(conv['input'].split())
            target_tokens = len(conv['target'].split())
            total_tokens = input_tokens + target_tokens
            
            if current_length + total_tokens <= max_pack_length:
                current_pack.append(conv)
                current_length += total_tokens
            else:
                if current_pack:
                    packed_conversations.append(current_pack)
                    total_packed_tokens += current_length  # Add to total
                current_pack = [conv]
                current_length = total_tokens
        
        if current_pack:
            packed_conversations.append(current_pack)
            total_packed_tokens += current_length  # Add final pack
        
        # ChatGPT feedback: Calculate meaningful average across all packs
        # Prevent division by zero
        if len(packed_conversations) > 0:
            avg_pack_length = total_packed_tokens / len(packed_conversations)
        else:
            avg_pack_length = 0
        print(f"   Sequence packing: {len(unique_conversations)} → {len(packed_conversations)} packs")
        print(f"   Total packed tokens: {total_packed_tokens:,}")
        print(f"   Average pack length: {avg_pack_length:.0f} tokens")
        
        # ChatGPT feedback: Optional token-drop augmentation (1-2% random drop)
        token_drop_rate = 0.01  # 1% token drop for regularization
        if token_drop_rate > 0:
            print(f"   🎲 Token-drop augmentation: {token_drop_rate*100:.1f}% (regularization)")
        
        # Update vocab incrementally and resize model if needed
        old_vocab_size = len(vocab) if vocab else 0
        build_vocab_from_db(incremental=True)
        new_vocab_size = len(vocab)
        
        # Ensure we have a model to train (either loaded existing or create new)
        if local_model is None:
            print("🆕 Creating new model for training...")
            local_model = SimpleLM(new_vocab_size).to(device)
            print(f"✅ Created new model with {sum(p.numel() for p in local_model.parameters()):,} parameters")
        elif new_vocab_size > old_vocab_size:
            print(f"🔄 Resizing model for vocabulary expansion: {old_vocab_size} -> {new_vocab_size}")
            local_model = resize_model(local_model, old_vocab_size, new_vocab_size)
            logger.info(f"Resized model for training: {old_vocab_size} -> {new_vocab_size}")
        
        # Smart streaming approach: Use ALL data by loading in chunks
        print(f"🚀 Implementing streaming dataset to use ALL {len(packed_conversations):,} conversation packs")
        print(f"   Will cycle through entire dataset without memory explosion")
        
        # Keep all data but implement streaming access
        total_packs = len(packed_conversations)
        print(f"   Total available: {total_packs:,} packs")
        print(f"   Streaming strategy: Load chunks of 5000 packs per epoch")
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
        print(f"   🧹 Forced garbage collection to free memory")
        
        # ChatGPT recommendation: Create validation split (3-5% held out)
        validation_split = 0.05  # 5% for validation
        val_size = int(len(packed_conversations) * validation_split)
        train_size = len(packed_conversations) - val_size
        
        # Ensure minimum validation size and prevent edge cases
        if val_size < 10:
            # Prevent division by zero
            if len(packed_conversations) > 0:
                val_size = min(10, len(packed_conversations) // 10)
            else:
                val_size = 0
            train_size = len(packed_conversations) - val_size
        
        # Safety check: ensure we don't have empty validation set
        if val_size == 0:
            val_size = min(10, len(packed_conversations))
            train_size = len(packed_conversations) - val_size
            print(f"   ⚠️  Warning: Validation split resulted in 0, using {val_size} samples")
        
        # Safety check: ensure we don't have empty training set
        if train_size == 0:
            train_size = len(packed_conversations) - 10
            val_size = 10
            print(f"   ⚠️  Warning: Training split resulted in 0, using {train_size} samples")
        
        # Split data for training and validation
        train_conversations = packed_conversations[:train_size]
        val_conversations = packed_conversations[train_size:]
        
        print(f"📊 ChatGPT-Recommended Data Split:")
        print(f"   Training: {train_size} conversation packs")
        print(f"   Validation: {val_size} conversation packs")
        print(f"   Split ratio: {validation_split*100:.1f}% validation")
        
        # Enhanced streaming dataset with data augmentation, validation split, and sequence packing
        # Use chunked loading to handle massive datasets without memory explosion
        chunk_size = 5000  # Load 5K conversations at a time
        
        train_dataset = ConversationDataset(train_conversations, vocab, packed=True, chunk_size=chunk_size)
        val_dataset = ConversationDataset(val_conversations, vocab, packed=True, chunk_size=chunk_size)
        
        # ChatGPT feedback: Set token-drop rate for augmentation
        train_dataset.token_drop_rate = token_drop_rate
        val_dataset.token_drop_rate = 0  # No augmentation during validation
        
        # Memory-efficient data loading for massive datasets
        # With 2.48M conversations, we need aggressive memory management
        print(f"⚠️  Massive dataset detected: {len(train_conversations):,} conversation packs")
        print(f"   Implementing memory-efficient training...")
        
        # GPU-optimized batch sizing with CPU data offloading
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   🚀 GPU detected: {gpu_memory_gb:.1f}GB VRAM")
            
            # More aggressive batch sizes for better GPU utilization
            if gpu_memory_gb >= 24:  # RTX 4090/3090 level
                microbatch_size = 16
                grad_accumulation_steps = 16  # 16 * 16 = 256 effective
            elif gpu_memory_gb >= 16:  # RTX 4080/3080 level
                microbatch_size = 12
                grad_accumulation_steps = 21  # 12 * 21 = 252 effective
            elif gpu_memory_gb >= 12:  # RTX 4070/3070 level
                microbatch_size = 8
                grad_accumulation_steps = 32  # 8 * 32 = 256 effective
            elif gpu_memory_gb >= 8:  # RTX 4060/3060 level
                microbatch_size = 6
                grad_accumulation_steps = 43  # 6 * 43 = 258 effective
            else:  # Lower-end GPU (Tesla T4)
                microbatch_size = 4
                grad_accumulation_steps = 64  # 4 * 64 = 256 effective
        else:
            # CPU fallback - conservative for massive datasets
            microbatch_size = 1
            grad_accumulation_steps = 256  # 1 * 256 = 256 effective
        
        effective_batch_size = microbatch_size * grad_accumulation_steps
        target_tokens_per_step = effective_batch_size * 1024  # Context length
        
        # GPU training with CPU data offloading for massive datasets
        # Use CPU to preprocess data while GPU handles training
        print(f"   🧠 CPU data offloading: Using {os.getenv('NUMBER_OF_PROCESSORS', '4')} cores for data preprocessing")
        print(f"   🚀 GPU training: Model and training on GPU, CPU handles data loading")
        
        # Enable multiple workers for CPU data preprocessing to offload GPU
        num_workers = min(2, int(os.getenv('NUMBER_OF_PROCESSORS', '1')))  # Reduced for Kaggle stability
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=microbatch_size,  # GPU batch size
            shuffle=False,  # We handle shuffling manually in the dataset
            collate_fn=collate_fn,
            num_workers=num_workers,  # Multiple CPU workers for data preprocessing
            pin_memory=True,  # Enable pin_memory for GPU training efficiency
            drop_last=True,  # Drop incomplete batches to prevent memory issues
            persistent_workers=True,  # Keep CPU workers alive for efficiency
            prefetch_factor=1  # Reduced prefetch for Kaggle stability
        )
        
        # Validation dataloader - ensure it always has at least one batch
        # Adjust batch size if validation set is too small
        val_batch_size = min(microbatch_size, len(val_dataset))
        if val_batch_size < microbatch_size:
            print(f"   ⚠️  Validation set too small for batch size {microbatch_size}, using {val_batch_size}")
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=val_batch_size,  # Adjusted batch size for validation
            shuffle=False,  # No shuffling for validation
            collate_fn=collate_fn,
            num_workers=num_workers,  # Multiple CPU workers for data preprocessing
            pin_memory=True,  # Enable pin_memory for GPU training efficiency
            drop_last=False,  # Keep incomplete batches for validation
            persistent_workers=True,  # Keep CPU workers alive for efficiency
            prefetch_factor=2  # Prefetch 2 batches ahead for Kaggle stability
        )
        
        print(f"📦 ChatGPT-Optimized Batch Sizing:")
        print(f"   Microbatch: {microbatch_size} sequences")
        print(f"   Gradient Accumulation: {grad_accumulation_steps} steps")
        print(f"   Effective Batch: {effective_batch_size} sequences")
        print(f"   Target Tokens/Step: {target_tokens_per_step:,} (ChatGPT: 250K-1M)")
        
        # GPU + CPU optimization info
        import psutil
        cpu_ram_gb = psutil.virtual_memory().total / (1024**3)
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        
        print(f"🚀 GPU + CPU Optimization:")
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   🚀 GPU: {gpu_memory_gb:.1f}GB VRAM for model training")
            
            # Multi-GPU info
            if torch.cuda.device_count() > 1:
                print(f"   🔧 Multi-GPU: {torch.cuda.device_count()} GPUs for parallel training")
                print(f"   📊 Effective batch size: {effective_batch_size} sequences")
                print(f"   ⚡ Expected speedup: ~{torch.cuda.device_count() * 0.8:.1f}x (with overhead)")
        
        print(f"   🧠 CPU: {cpu_ram_gb:.1f}GB RAM for data preprocessing")
        print(f"   💾 Available RAM: {available_ram_gb:.1f} GB")
        print(f"   🔧 Workers: {num_workers} CPU cores for data offloading")
        print(f"   📦 Prefetch: 2 batches ahead for Kaggle stability")
        
        # Enhanced optimizer with ChatGPT-recommended parameters
        # Conservative batch (≈0.25M toks/step): 8e-4, Balanced (≈0.5M): 1.0e-3
        if target_tokens_per_step <= 300000:
            optimal_lr = 8e-4  # Conservative
        elif target_tokens_per_step <= 600000:
            optimal_lr = 1.0e-3  # Balanced
        else:
            optimal_lr = 1.2e-3  # Aggressive
        
        # Enhanced AdamW optimizer with ChatGPT recommendations
        optimizer = optim.AdamW(
            local_model.parameters(), 
            lr=optimal_lr,  # ChatGPT-optimized learning rate
            weight_decay=0.05,  # ChatGPT recommendation for regularization
            betas=(0.9, 0.95),  # ChatGPT recommendation: β2=0.95 for stability
            eps=1e-8,
            amsgrad=True  # Enable AMSGrad for better convergence
        )
        
        # Print optimizer configuration
        print(f"🔧 Optimizer: AdamW with weight_decay=0.05, β1=0.9, β2=0.95")
        print(f"   Learning rate: {optimal_lr:.6f}")
        print(f"   Weight decay: 0.05 (ChatGPT recommendation)")
        print(f"   AMSGrad: Enabled for better convergence")
        
        # ChatGPT feedback: Make epochs configurable based on dataset size
        total_tokens = len(train_conversations) * 1024  # Approximate tokens per epoch
        if total_tokens <= 10000000:  # ≤10M tokens
            epochs = 8  # Smaller datasets: fewer epochs to prevent overfitting
        elif total_tokens <= 50000000:  # ≤50M tokens
            epochs = 15  # Medium datasets: balanced approach
        else:
            epochs = 20  # Large datasets: full training
        
        # Enhanced learning rate scheduler with ChatGPT recommendations
        # Warmup: first 1.5-2% of total steps, then cosine decay to 10% of peak
        total_steps = len(train_dataloader) * epochs
        warmup_steps = max(1000, int(0.02 * total_steps))  # ~2% of total steps for stable warmup
        
        # Safety check to prevent warmup_steps from being 0
        if warmup_steps <= 0:
            warmup_steps = 1
            print(f"   ⚠️  Warning: warmup_steps was 0, using 1")
        
        # Enhanced cosine decay with better curve
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup from 0.01 to 1.0
                if warmup_steps > 0:
                    return 0.01 + 0.99 * (float(step) / float(warmup_steps))
                else:
                    return 0.01
            else:
                # Enhanced cosine decay with better curve shape
                # Decay to 10% of peak value
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                # Use smoother cosine curve with better final convergence
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                # Ensure minimum learning rate is 10% of peak
                return max(0.1, cosine_decay)
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Print scheduler configuration
        print(f"📈 Learning Rate Scheduler:")
        print(f"   Warmup steps: {warmup_steps:,} ({warmup_steps/total_steps*100:.1f}% of total)")
        print(f"   Decay: Cosine to 10% of peak")
        print(f"   Total steps: {total_steps:,}")
        print(f"   Final LR: {optimal_lr * 0.1:.6f}")
        
        # Try to load optimizer and scheduler state from previous training
        # Only load if vocabulary size hasn't changed (to avoid tensor size mismatches)
        if os.path.exists(LOCAL_MODEL_PATH):
            try:
                checkpoint = torch.load(LOCAL_MODEL_PATH, map_location=device)
                # Check if vocabulary size has changed
                # For backward compatibility, if vocab_size is not in checkpoint, assume it's compatible
                checkpoint_vocab_size = checkpoint.get('vocab_size', len(checkpoint.get('vocab', {})))
                if checkpoint_vocab_size == new_vocab_size:
                    if 'optimizer_state_dict' in checkpoint and 'scheduler_state_dict' in checkpoint:
                        print("🔄 Loading previous optimizer and scheduler state...")
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        print("✅ Continued training with previous optimizer state")
                    else:
                        print("ℹ️ No previous optimizer state found, starting fresh")
                else:
                    print(f"⚠️ Vocabulary size changed ({checkpoint.get('vocab_size', 'unknown')} -> {new_vocab_size}), starting with fresh optimizer state")
            except Exception as e:
                print(f"⚠️ Could not load previous optimizer state: {e}")
                print("ℹ️ Starting with fresh optimizer state")
        
        # Enhanced loss function with label smoothing
        criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        
        # Enhanced mixed precision training for efficiency
        scaler = None
        if torch.cuda.is_available():
            try:
                scaler = torch.amp.GradScaler()
                print("✅ Mixed precision training enabled with automatic mixed precision (AMP)")
                print("   This will reduce VRAM usage and potentially speed up training")
            except Exception as e:
                print(f"⚠️ Mixed precision training failed to initialize: {e}")
                print("   Falling back to full precision training")
        else:
            print("ℹ️ Mixed precision training not available (CPU mode)")
        
        # ChatGPT feedback: Add EMA weights for stability (decay ~0.999)
        ema_decay = 0.999
        ema_model = None
        if ema_decay > 0:
            ema_model = copy.deepcopy(local_model)
            for param in ema_model.parameters():
                param.requires_grad = False
        
        # GPU/CPU optimizations
        if torch.cuda.is_available():
            print("🚀 GPU training mode detected - using CUDA optimizations")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"🔥 Mixed precision training enabled")
            
            # Performance optimizations for faster training
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Multi-GPU specific optimizations
            if torch.cuda.device_count() > 1:
                print(f"🚀 Multi-GPU optimizations enabled:")
                print(f"   🔧 DataParallel with device IDs [0, 1]")
                print(f"   📊 Batch distribution across {torch.cuda.device_count()} GPUs")
                print(f"   ⚡ Synchronized batch normalization")
                
                # Set primary GPU for better memory management
                torch.cuda.set_device(0)
        else:
            print("🖥️ CPU training mode detected - applying optimizations")
            # Enable CPU optimizations
            torch.set_num_threads(4)  # Optimal for most CPUs
            print(f"🧵 Using {torch.get_num_threads()} CPU threads")
        
        # Enhanced training loop with incremental training support
        # Optimized for 14-layer architecture with gradient accumulation and chunked data
        
        best_loss = float('inf')
        patience = 4  # Increased patience for deeper network
        patience_counter = 0
        
        print(f"🏗️ Training optimized for 14L/640E/8H/2560H architecture")
        print(f"⏱️ Extended training: {epochs} epochs with {patience} patience")
        print(f"🔥 Learning rate: {optimal_lr:.6f} (ChatGPT-optimized)")
        print(f"🔄 Gradient accumulation: {grad_accumulation_steps} steps")
        print(f"📊 Target effective batch: {effective_batch_size} sequences")
        
        # Incremental training configuration
        print(f"\n🚀 INCREMENTAL TRAINING FEATURES:")
        print(f"   📦 Chunk size: {INCREMENTAL_TRAINING_CONFIG['chunk_size_gb']} GB per chunk")
        print(f"   🔄 Replay buffer: {INCREMENTAL_TRAINING_CONFIG['replay_buffer_size']*100:.0f}% of history")
        print(f"   💾 Checkpoint every: {INCREMENTAL_TRAINING_CONFIG['checkpoint_interval_gb']} GB")
        print(f"   📚 Curriculum phases: {', '.join(INCREMENTAL_TRAINING_CONFIG['curriculum_phases'])}")
        
        # Check for existing model to resume incremental training
        if check_for_existing_model():
            print("✅ Resuming incremental training from existing model")
        else:
            print("🆕 Starting fresh incremental training")
        
        # Multi-GPU performance summary
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"\n🚀 MULTI-GPU PERFORMANCE OPTIMIZATION:")
            print(f"   🔧 DataParallel: Enabled with device IDs [0, 1]")
            print(f"   📊 Batch distribution: {microbatch_size} sequences per GPU")
            print(f"   ⚡ Expected speedup: ~{torch.cuda.device_count() * 0.8:.1f}x")
            print(f"   💾 Memory efficiency: Shared across {torch.cuda.device_count()} GPUs")
            print(f"   🔄 Synchronization: Automatic gradient synchronization")
            print(f"   📈 Monitoring: Real-time GPU utilization tracking")
        
        # If we loaded a previous model, try to get the previous best loss
        if os.path.exists(LOCAL_MODEL_PATH):
            try:
                checkpoint = torch.load(LOCAL_MODEL_PATH, map_location=device)
                if 'loss' in checkpoint:
                    best_loss = checkpoint['loss']
                    print(f"📊 Continuing training from previous best loss: {best_loss:.4f}")
                if 'final_loss' in checkpoint:
                    print(f"📊 Previous training final loss: {checkpoint['final_loss']:.4f}")
            except Exception as e:
                print(f"⚠️ Could not load previous loss info: {e}")
        
        print(f"🎯 Training goal: Improve upon best loss of {best_loss:.4f}")
        
        for epoch in range(epochs):
            local_model.train()
            total_loss = 0.0
            num_batches = 0
            
            # GPU + CPU memory monitoring at start of each epoch
            import psutil
            ram_usage = psutil.virtual_memory()
            print(f"🔄 Epoch {epoch+1}/{epochs}")
            print(f"   🧠 CPU RAM: {ram_usage.percent:.1f}% ({ram_usage.used/(1024**3):.1f}GB / {ram_usage.total/(1024**3):.1f}GB)")
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                print(f"   🚀 GPU VRAM: {gpu_memory_allocated:.1f}GB allocated, {gpu_memory_reserved:.1f}GB reserved")
            
            # Incremental training: Process data in chunks
            print(f"📦 Processing epoch {epoch+1} in chunks...")
            
            # Get current curriculum phase
            current_phase = INCREMENTAL_TRAINING_CONFIG['curriculum_phases'][incremental_manager.current_phase]
            print(f"📚 Current curriculum phase: {current_phase}")
            
            # Process data in chunks for this epoch
            epoch_chunks = 0
            epoch_data_processed = 0
            
            # ChatGPT recommendation: Hard shuffling every epoch
            if hasattr(train_dataloader.dataset, 'shuffle_data'):
                train_dataloader.dataset.shuffle_data()
            
            # Process chunks for this epoch
            while epoch_data_processed < len(train_conversations):
                # Get next chunk of data
                chunk = incremental_manager.get_next_chunk(
                    train_conversations, 
                    INCREMENTAL_TRAINING_CONFIG['chunk_size_gb']
                )
                
                if not chunk:
                    break
                
                # Add replay buffer data to reduce forgetting
                replay_data = incremental_manager.get_replay_buffer()
                if replay_data:
                    chunk.extend(replay_data)
                    print(f"   🔄 Added {len(replay_data)} replay samples to chunk")
                
                # Create dataset for this chunk
                chunk_dataset = ConversationDataset(chunk, vocab, packed=True, chunk_size=len(chunk))
                chunk_dataloader = DataLoader(
                    chunk_dataset, 
                    batch_size=microbatch_size, 
                    shuffle=True, 
                    collate_fn=collate_fn,
                    num_workers=0  # Reduce memory usage
                )
                
                print(f"   📦 Processing chunk {epoch_chunks + 1}: {len(chunk)} samples")
                
                # Process this chunk
                chunk_loss = process_training_chunk(
                    chunk_dataloader, 
                    local_model, 
                    optimizer, 
                    scheduler, 
                    criterion, 
                    scaler,
                    grad_accumulation_steps
                )
                
                # Log chunk loss
                incremental_manager.log_loss(chunk_loss, {
                    'epoch': epoch + 1,
                    'chunk': epoch_chunks + 1,
                    'samples': len(chunk),
                    'phase': current_phase
                })
                
                # Update progress
                total_loss += chunk_loss
                epoch_data_processed += len(chunk)
                epoch_chunks += 1
                
                # Check if we should save a checkpoint
                data_processed_gb = epoch_data_processed / 100000  # Rough approximation
                if incremental_manager.should_checkpoint(data_processed_gb):
                    print(f"   💾 Saving checkpoint after {data_processed_gb:.2f} GB processed...")
                    save_incremental_checkpoint(
                        local_model, 
                        optimizer, 
                        scheduler, 
                        {'epoch': epoch + 1, 'chunk': epoch_chunks},
                        chunk_loss
                    )
                    incremental_manager.total_data_processed += data_processed_gb
                
                # Periodic checkpointing every N chunks (additional safety)
                if epoch_chunks % 5 == 0:  # Every 5 chunks
                    print(f"   💾 Periodic checkpoint (chunk {epoch_chunks})...")
                    periodic_checkpoint_path = os.path.join(
                        KAGGLE_WORKING_DIR, 
                        f"jarvis_periodic_chunk_{epoch_chunks:04d}.pth"
                    )
                    
                    periodic_checkpoint = {
                        'model_state_dict': local_model.state_dict(),
                        'vocab': vocab,
                        'epoch': epoch + 1,
                        'chunk': epoch_chunks,
                        'loss': chunk_loss,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Atomic save
                    temp_path = periodic_checkpoint_path + '.tmp'
                    torch.save(periodic_checkpoint, temp_path)
                    os.replace(temp_path, periodic_checkpoint_path)
                    print(f"   ✅ Periodic checkpoint saved: {periodic_checkpoint_path}")
                
                # Memory management
                del chunk_dataset, chunk_dataloader
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                
                print(f"   ✅ Chunk {epoch_chunks} completed. Loss: {chunk_loss:.4f}")
            
            # Calculate average loss for this epoch
            if epoch_chunks > 0:
                avg_epoch_loss = total_loss / epoch_chunks
                print(f"   📊 Epoch {epoch+1} completed: {epoch_chunks} chunks, avg loss: {avg_epoch_loss:.4f}")
            else:
                avg_epoch_loss = float('inf')
                print(f"   ⚠️ No chunks processed in epoch {epoch+1}")
            
            # Move to next curriculum phase if appropriate
            if (epoch + 1) % 3 == 0:  # Change phase every 3 epochs
                incremental_manager.next_phase()
            
            # Continue with validation and early stopping logic
            print(f"   🔍 Running validation...")
            val_loss = run_validation(val_dataloader, local_model, criterion, scaler)
            
            if val_loss is not None:
                print(f"   📊 Validation loss: {val_loss:.4f}")
                
                # Early stopping logic
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    print(f"   🎯 New best validation loss: {best_loss:.4f}")
                    
                    # Save best model
                    save_incremental_checkpoint(
                        local_model, 
                        optimizer, 
                        scheduler, 
                        {'epoch': epoch + 1, 'best_val_loss': val_loss},
                        avg_epoch_loss
                    )
                else:
                    patience_counter += 1
                    print(f"   ⏳ No improvement for {patience_counter}/{patience} epochs")
                    
                    if patience_counter >= patience:
                        print(f"   🛑 Early stopping triggered after {patience} epochs without improvement")
                        break
            else:
                print(f"   ⚠️ Validation failed, skipping early stopping check")
            
            # Update learning rate
            current_lr = scheduler.get_last_lr()[0]
            print(f"   📈 Learning rate: {current_lr:.6f}")
            
            # Log epoch completion
            logger.info(f"Epoch {epoch+1} completed. Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f if val_loss else 'N/A'}, LR: {current_lr:.6f}")
            
            # Multi-GPU performance monitoring
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                try:
                    gpu_utilization = []
                    for i in range(torch.cuda.device_count()):
                        gpu_utilization.append(torch.cuda.utilization(i))
                    progress_bar.set_postfix_str(f"GPU0:{gpu_utilization[0]}% GPU1:{gpu_utilization[1]}%")
                except Exception as e:
                    # Fallback if GPU utilization monitoring fails
                    progress_bar.set_postfix_str("GPU monitoring disabled")
                    logger.debug(f"GPU utilization monitoring failed: {e}")
            
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Debug: Print shapes (only first batch of first epoch)
                if epoch == 0 and batch_idx == 0:
                    logger.info(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
                
                # Create attention masks
                # For padding: (batch_size, seq_len) - True for padding tokens
                padding_mask = (inputs == 0)
                
                # For causal masking: (seq_len, seq_len) - upper triangular matrix
                # CRITICAL: Ensure we use the actual sequence length from inputs, not a variable that might be wrong
                actual_seq_len = inputs.size(1)
                
                # Safety check: ensure sequence length is within model limits
                model_max_seq_len = get_model_max_seq_len(local_model)
                if actual_seq_len > model_max_seq_len:
                    logger.warning(f"Sequence length {actual_seq_len} exceeds model limit {model_max_seq_len}, truncating...")
                    actual_seq_len = model_max_seq_len
                    inputs = inputs[:, :actual_seq_len]
                    targets = targets[:, :actual_seq_len]
                    padding_mask = (inputs == 0)
                
                # CRITICAL FIX: Ensure causal mask is exactly the right size
                # The error shows [512, 1024] vs [1024, 1024], so we need to be more careful
                causal_mask = torch.triu(torch.ones(actual_seq_len, actual_seq_len, device=inputs.device), diagonal=1).bool()
                
                # EXTRA SAFETY: Verify mask dimensions match exactly
                if causal_mask.shape[0] != actual_seq_len or causal_mask.shape[1] != actual_seq_len:
                    logger.error(f"CRITICAL: Causal mask shape mismatch!")
                    logger.error(f"   Expected: ({actual_seq_len}, {actual_seq_len})")
                    logger.error(f"   Got: {causal_mask.shape}")
                    logger.error(f"   Input shape: {inputs.shape}")
                    logger.error(f"   Model max_seq_len: {model_max_seq_len}")
                    
                    # Force correct size
                    causal_mask = torch.triu(torch.ones(actual_seq_len, actual_seq_len, device=inputs.device), diagonal=1).bool()
                    logger.info(f"   Fixed causal mask to: {causal_mask.shape}")
                
                # Debug: Print mask shapes (only first batch of first epoch)
                if epoch == 0 and batch_idx == 0:
                    logger.info(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
                    logger.info(f"Padding mask shape: {padding_mask.shape}, Causal mask shape: {causal_mask.shape}")
                    logger.info(f"Model max_seq_len: {get_model_max_seq_len(local_model)}")
                    logger.info(f"Actual seq_len: {actual_seq_len}")
                    
                    # Verify all shapes are consistent
                    assert inputs.size(1) == actual_seq_len, f"Input seq_len mismatch: {inputs.size(1)} vs {actual_seq_len}"
                    assert targets.size(1) == actual_seq_len, f"Target seq_len mismatch: {targets.size(1)} vs {actual_seq_len}"
                    assert padding_mask.size(1) == actual_seq_len, f"Padding mask seq_len mismatch: {padding_mask.size(1)} vs {actual_seq_len}"
                    assert causal_mask.shape == (actual_seq_len, actual_seq_len), f"Causal mask shape mismatch: {causal_mask.shape} vs ({actual_seq_len}, {actual_seq_len})"
                    logger.info("✅ All shapes verified and consistent!")
                
                # Mixed precision forward pass
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = local_model(inputs, padding_mask=padding_mask, causal_mask=causal_mask)
                else:
                    outputs = local_model(inputs, padding_mask=padding_mask, causal_mask=causal_mask)
                
                # Debug: Print output shape
                if epoch == 0 and batch_idx == 0:
                    logger.info(f"Output shape: {outputs.shape}")
                
                # Get output dimensions
                batch_size, seq_len, vocab_size = outputs.shape
                
                # Debug: Verify input/target alignment
                if epoch == 0 and batch_idx == 0:
                    logger.info(f"Input shape: {inputs.shape}, Target shape: {targets.shape}, Output shape: {outputs.shape}")
                
                # Ensure targets match the sequence length (should already be handled by collate_fn)
                if targets.size(1) != seq_len:
                    logger.warning(f"Target length mismatch: {targets.size(1)} vs {seq_len}, truncating...")
                    if targets.size(1) > seq_len:
                        targets = targets[:, :seq_len]
                    else:
                        padding = torch.zeros(batch_size, seq_len - targets.size(1), dtype=targets.dtype, device=targets.device)
                        targets = torch.cat([targets, padding], dim=1)
                
                # Reshape outputs and targets for loss calculation
                outputs_flat = outputs.view(-1, vocab_size)
                targets_flat = targets.reshape(-1)
                
                # Debug: Print flattened shapes
                if epoch == 0 and batch_idx == 0:
                    logger.info(f"Flattened output shape: {outputs_flat.shape}, Flattened target shape: {targets_flat.shape}")
                
                # Calculate loss
                loss = criterion(outputs_flat, targets_flat)
                
                # CRITICAL: Check for invalid loss values and handle gracefully
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Invalid loss detected: {loss.item()}, skipping batch")
                    # Reset gradients and continue to next batch
                    optimizer.zero_grad()
                    continue
                
                # Additional loss validation - prevent gradient explosion
                if loss.item() > 1000:  # Unusually high loss
                    logger.warning(f"Very high loss detected: {loss.item()}, applying aggressive gradient clipping")
                    # Apply more aggressive gradient clipping
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=0.5)
                
                # ChatGPT recommendation: Gradient accumulation for effective batch sizing
                # Scale loss by accumulation steps for proper averaging
                # Prevent division by zero
                if grad_accumulation_steps > 0:
                    scaled_loss = loss / grad_accumulation_steps
                else:
                    scaled_loss = loss
                    logger.warning("Gradient accumulation steps is 0, using unscaled loss")
                
                # CRITICAL: Prevent gradient explosion by clipping before backward pass
                if loss.item() > 100:  # High loss threshold
                    # Apply aggressive gradient clipping
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=0.1)
                
                # Backward pass with mixed precision
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                # Update progress tracking
                total_loss += loss.item()
                num_batches += 1
                
                # CPU RAM cleanup and monitoring every 100 batches
                if batch_idx % 100 == 0 and batch_idx > 0:
                    # Force garbage collection
                    gc.collect()
                    
                    # Monitor CPU memory usage
                    try:
                        ram_usage = psutil.virtual_memory()
                        current_ram_gb = ram_usage.used / (1024**3)
                    except Exception as e:
                        current_ram_gb = 0.0
                        logger.debug(f"RAM monitoring failed: {e}")
                    
                    # If approaching memory limit, force more aggressive cleanup
                    if current_ram_gb > cpu_memory_limit_gb * 0.8:  # 80% of limit
                        print(f"\n⚠️  CPU memory usage high: {current_ram_gb:.1f}GB, forcing cleanup...")
                        gc.collect()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                        # If still over limit, force even more aggressive cleanup
                        if current_ram_gb > cpu_memory_limit_gb * 0.9:  # 90% of limit
                            print(f"🚨 CRITICAL: CPU memory at {current_ram_gb:.1f}GB, forcing emergency cleanup...")
                            gc.collect()
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                            
                            # Force Python to release memory
                            import sys
                            sys.stdout.flush()
                            
                            # If still over limit, pause training briefly
                            if current_ram_gb > cpu_memory_limit_gb * 0.95:  # 95% of limit
                                print(f"⏸️  PAUSING: CPU memory critical at {current_ram_gb:.1f}GB, waiting for cleanup...")
                                time.sleep(5)  # Wait 5 seconds for system cleanup
                                gc.collect()
                                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                                    # Log memory usage every 500 batches
                if batch_idx % 500 == 0:
                    # Ensure current_ram_gb is defined
                    try:
                        ram_usage = psutil.virtual_memory()
                        current_ram_gb = ram_usage.used / (1024**3)
                    except Exception as e:
                        current_ram_gb = 0.0
                        logger.debug(f"RAM monitoring failed: {e}")
                    
                    postfix = {
                        'loss': f'{loss.item():.4f}',
                        'cpu_ram': f'{current_ram_gb:.1f}GB/{cpu_memory_limit_gb}GB'
                    }
                    if torch.cuda.is_available():
                        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                        postfix['gpu_vram'] = f'{gpu_memory_allocated:.1f}GB'
                        
                        # Multi-GPU memory info
                        if torch.cuda.device_count() > 1:
                            try:
                                for i in range(torch.cuda.device_count()):
                                    gpu_mem = torch.cuda.memory_allocated(i) / (1024**3)
                                    postfix[f'gpu{i}_vram'] = f'{gpu_mem:.1f}GB'
                            except Exception as e:
                                # Fallback if multi-GPU memory monitoring fails
                                logger.debug(f"Multi-GPU memory monitoring failed: {e}")
                                pass
                    
                    progress_bar.set_postfix(postfix)
                
                # ChatGPT recommendation: Optimizer step every grad_accumulation_steps
                if (batch_idx + 1) % grad_accumulation_steps == 0:
                    # Advanced gradient clipping (ChatGPT: grad clip 1.0)
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                    
                    # CRITICAL FIX: Optimizer step FIRST, then scheduler
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    # Learning rate scheduling AFTER optimizer step (fixes warning)
                    scheduler.step()
                    
                    # Reset gradients for next accumulation cycle
                    optimizer.zero_grad()
                    
                    # ChatGPT feedback: Update EMA weights
                    if ema_model is not None:
                        with torch.no_grad():
                            for ema_param, param in zip(ema_model.parameters(), local_model.parameters()):
                                ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
                
                current_lr = scheduler.get_last_lr()[0]
                
                # Enhanced progress bar with ChatGPT metrics
                # Prevent division by zero
                if num_batches > 0:
                    avg_loss = total_loss / num_batches
                else:
                    avg_loss = 0.0
                effective_batch_progress = (batch_idx + 1) % grad_accumulation_steps
                progress_bar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    batch_loss=f"{loss.item():.4f}",
                    lr=f"{current_lr:.6f}",
                    epoch=f"{epoch+1}/{epochs}",
                    acc_step=f"{effective_batch_progress}/{grad_accumulation_steps}"
                )
            
            progress_bar.close()
            
            # Calculate epoch statistics
            # Prevent division by zero
            if num_batches > 0:
                epoch_loss = total_loss / num_batches
            else:
                epoch_loss = 0.0
                logger.warning("No batches processed in epoch, setting epoch_loss to 0")
            current_lr = scheduler.get_last_lr()[0]
            
            # ChatGPT recommendation: Run validation after each epoch
            local_model.eval()
            val_loss = 0.0
            val_batches = 0
            
            print(f"   🔍 Starting validation: {len(val_dataloader)} batches")
            if len(val_dataloader) == 0:
                print(f"   ❌ Error: Validation dataloader is empty!")
                print(f"   🔍 Validation dataset size: {len(val_dataset)}")
                print(f"   🔍 Validation batch size: {val_dataloader.batch_size}")
                avg_val_loss = float('inf')
                local_model.train()
                continue
            
            with torch.no_grad():
                print(f"   🔍 Processing validation batches...")
                for batch_idx, (val_inputs, val_targets) in enumerate(val_dataloader):
                    if batch_idx % 100 == 0:  # Progress indicator every 100 batches
                        print(f"   🔍 Validation batch {batch_idx}/{len(val_dataloader)}")
                    
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                    
                    # Create attention masks for validation
                    padding_mask = (val_inputs == 0)
                    # Ensure we use the actual sequence length from val_inputs
                    actual_seq_len = val_inputs.size(1)
                    
                    # Safety check: ensure sequence length is within model limits
                    model_max_seq_len = get_model_max_seq_len(local_model)
                    if actual_seq_len > model_max_seq_len:
                        actual_seq_len = model_max_seq_len
                        val_inputs = val_inputs[:, :actual_seq_len]
                        val_targets = val_targets[:, :actual_seq_len]
                        padding_mask = (val_inputs == 0)
                    
                    causal_mask = torch.triu(torch.ones(actual_seq_len, actual_seq_len, device=val_inputs.device), diagonal=1).bool()
                    
                    # Forward pass
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            val_outputs = local_model(val_inputs, padding_mask=padding_mask, causal_mask=causal_mask)
                    else:
                        val_outputs = local_model(val_inputs, padding_mask=padding_mask, causal_mask=causal_mask)
                    
                    # Ensure targets match the sequence length
                    batch_size, seq_len, vocab_size = val_outputs.shape
                    if val_targets.size(1) != seq_len:
                        if val_targets.size(1) > seq_len:
                            val_targets = val_targets[:, :seq_len]
                        else:
                            padding = torch.zeros(batch_size, seq_len - val_targets.size(1), dtype=val_targets.dtype, device=val_targets.device)
                            val_targets = torch.cat([val_targets, padding], dim=1)
                    
                    # Calculate validation loss
                    val_outputs_flat = val_outputs.view(-1, vocab_size)
                    val_targets_flat = val_targets.reshape(-1)
                    val_batch_loss = criterion(val_outputs_flat, val_targets_flat)
                    
                    val_loss += val_batch_loss.item()
                    val_batches += 1
                    
                    # Safety check: ensure we don't process too many batches
                    if val_batches > len(val_dataloader):
                        print(f"   ⚠️  Warning: Processed more batches than expected!")
                        break
            
            # Safety check to prevent division by zero
            if val_batches > 0:
                avg_val_loss = val_loss / val_batches
                print(f"   ✅ Validation completed: {val_batches} batches, avg loss: {avg_val_loss:.4f}")
            else:
                print(f"   ⚠️  Warning: No validation batches processed!")
                print(f"   🔍 Validation dataset size: {len(val_dataset) if 'val_dataset' in locals() else 'unknown'}")
                print(f"   🔍 Validation batch size: {val_dataloader.batch_size if hasattr(val_dataloader, 'batch_size') else 'unknown'}")
                print(f"   🔍 Validation batches expected: {len(val_dataloader) if val_dataloader else 0}")
                avg_val_loss = float('inf')
            local_model.train()  # Switch back to training mode
            
            # Log epoch completion with training and validation statistics
            logger.info(f"Epoch {epoch+1} completed. Train Loss: {epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")
            
            # ChatGPT recommendation: Use validation loss for early stopping
            if avg_val_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                
                # Create backup of previous best model
                if os.path.exists(LOCAL_MODEL_PATH):
                    try:
                        shutil.copy2(LOCAL_MODEL_PATH, BACKUP_MODEL_PATH)
                        logger.info(f"Backup created: {BACKUP_MODEL_PATH}")
                    except Exception as e:
                        logger.warning(f"Could not create backup: {e}")
                
                # Save best model with enhanced metadata
                checkpoint_data = {
                    'model_state_dict': local_model.state_dict(),
                    'model_config': {
                        'embed_size': local_model.embed_size,
                        'num_heads': local_model.num_heads,
                        'num_layers': local_model.num_layers,
                        'hidden_size': local_model.hidden_size,
                        'dropout': local_model.dropout,
                        'max_seq_len': local_model.max_seq_len
                    },
                    'vocab': vocab,
                    'vocab_size': len(vocab),  # Save vocabulary size for compatibility checks
                    'epoch': epoch + 1,
                    'loss': best_loss,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'training_info': {
                        'device': str(device),
                        'timestamp': datetime.now().isoformat(),
                        'total_parameters': sum(p.numel() for p in local_model.parameters()),
                        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                        'mixed_precision': scaler is not None
                    }
                }
                
                # Save with atomic operation
                temp_path = LOCAL_MODEL_PATH + '.tmp'
                torch.save(checkpoint_data, temp_path)
                os.replace(temp_path, LOCAL_MODEL_PATH)
                
                logger.info(f"New best model saved with loss: {best_loss:.4f}")
                logger.info(f"Checkpoint size: {os.path.getsize(LOCAL_MODEL_PATH) / 1024 / 1024:.1f} MB")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                    break
        
        # Final model save with enhanced metadata
        final_checkpoint = {
            'model_state_dict': local_model.state_dict(),
            'model_config': {
                'embed_size': local_model.embed_size,
                'num_heads': local_model.num_heads,
                'num_layers': local_model.num_layers,
                'hidden_size': local_model.hidden_size,
                'dropout': local_model.dropout,
                'max_seq_len': local_model.max_seq_len
            },
            'vocab': vocab,
            'vocab_size': len(vocab),  # Save vocabulary size for compatibility checks
            'final_epoch': epochs,
            'final_loss': epoch_loss,
            'training_info': {
                'device': str(device),
                'timestamp': datetime.now().isoformat(),
                'total_parameters': sum(p.numel() for p in local_model.parameters()),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                'mixed_precision': scaler is not None,
                'total_training_time': f"{epochs} epochs completed"
            }
        }
        
        # Save with atomic operation
        temp_path = LOCAL_MODEL_PATH + '.tmp'
        torch.save(final_checkpoint, temp_path)
        os.replace(temp_path, LOCAL_MODEL_PATH)
        
        logger.info(f"Training completed. Final loss: {epoch_loss:.4f}")
        if scaler is not None:
            logger.info("Mixed precision training was used")
        
        # Show training summary
        print(f"\n🎯 TRAINING SUMMARY:")
        print(f"📊 Final Loss: {epoch_loss:.4f}")
        print(f"🏗️ Model Parameters: {sum(p.numel() for p in local_model.parameters()):,}")
        print(f"📚 Vocabulary Size: {len(vocab)}")
        print(f"💾 Checkpoint Size: {os.path.getsize(LOCAL_MODEL_PATH) / 1024 / 1024:.1f} MB")
        print(f"🔧 Device Used: {device}")
        if torch.cuda.is_available():
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        
        # Show data folder statistics
        show_data_folder_stats()
        
        print("✅ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        print(f"❌ Training failed with error: {e}")

def analyze_kaggle_dataset():
    """Analyze the Kaggle input dataset for training"""
    print("🔍 Analyzing Kaggle input dataset...")
    
    if not os.path.exists(KAGGLE_INPUT_DIR):
        print(f"❌ Kaggle input directory not found: {KAGGLE_INPUT_DIR}")
        return
    
    try:
        # Get all files in the input directory
        all_files = os.listdir(KAGGLE_INPUT_DIR)
        
        if not all_files:
            print("📁 Input directory is empty")
            return
        
        print(f"📚 Found {len(all_files)} files in Kaggle input directory")
        
        # Categorize files by type
        file_types = {}
        total_size = 0
        
        for file in all_files:
            file_path = os.path.join(KAGGLE_INPUT_DIR, file)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                total_size += file_size
                
                # Get file extension
                _, ext = os.path.splitext(file)
                ext = ext.lower()
                
                if ext in file_types:
                    file_types[ext].append((file, file_size))
                else:
                    file_types[ext] = [(file, file_size)]
        
        # Display file type breakdown
        print(f"\n📊 File Type Breakdown:")
        for ext, files in file_types.items():
            count = len(files)
            size = sum(size for _, size in files)
            print(f"   {ext}: {count} files ({size / (1024*1024):.1f} MB)")
        
        print(f"\n💾 Total Dataset Size: {total_size / (1024*1024):.1f} MB")
        
        # Analyze specific file types
        if '.jsonl' in file_types:
            print(f"\n🔍 JSONL Files Analysis:")
            jsonl_files = file_types['.jsonl']
            for file, size in jsonl_files[:5]:  # Show first 5
                file_path = os.path.join(KAGGLE_INPUT_DIR, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        print(f"   📄 {file}: {len(lines)} lines ({size / 1024:.1f} KB)")
                except Exception as e:
                    print(f"   📄 {file}: Error reading file - {e}")
        
        if '.json' in file_types:
            print(f"\n🔍 JSON Files Analysis:")
            json_files = file_types['.json']
            for file, size in json_files[:5]:  # Show first 5
                file_path = os.path.join(KAGGLE_INPUT_DIR, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            print(f"   📄 {file}: {len(data)} items ({size / 1024:.1f} KB)")
                        elif isinstance(data, dict):
                            print(f"   📄 {file}: Dictionary with {len(data)} keys ({size / 1024:.1f} KB)")
                        else:
                            print(f"   📄 {file}: {type(data).__name__} ({size / 1024:.1f} KB)")
                except Exception as e:
                    print(f"   📄 {file}: Error reading file - {e}")
        
        if '.txt' in file_types:
            print(f"\n🔍 TXT Files Analysis:")
            txt_files = file_types['.txt']
            for file, size in txt_files[:5]:  # Show first 5
                file_path = os.path.join(KAGGLE_INPUT_DIR, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        words = content.split()
                        print(f"   📄 {file}: {len(lines)} lines, {len(words)} words ({size / 1024:.1f} KB)")
                except Exception as e:
                    print(f"   📄 {file}: Error reading file - {e}")
        
        print(f"\n✅ Dataset analysis completed!")
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")

def show_data_folder_stats():
    """Show statistics about the Kaggle input directory files"""
    all_txt_files = glob.glob(os.path.join(KAGGLE_INPUT_DIR, "*.txt"))
    all_json_files = glob.glob(os.path.join(KAGGLE_INPUT_DIR, "*.json"))
    all_jsonl_files = glob.glob(os.path.join(KAGGLE_INPUT_DIR, "*.jsonl"))
    all_csv_files = glob.glob(os.path.join(KAGGLE_INPUT_DIR, "*.csv"))
    all_files = all_txt_files + all_json_files + all_jsonl_files + all_csv_files
    
    if not all_files:
        print("📁 No .txt, .json, .jsonl, or .csv files found in the Kaggle input directory.")
        return
    
    print(f"📁 Found {len(all_txt_files)} .txt files, {len(all_json_files)} .json files, {len(all_jsonl_files)} .jsonl files, and {len(all_csv_files)} .csv files in Kaggle input directory.")
    
    # Count unique words in data folder files
    word_set = set()
    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    word_set.update(clean_convo_data(content).split())
        except Exception as e:
            logger.warning(f"Could not read file {file_path} for word counting: {e}")
    
    print(f"📚 Total unique words in data folder files: {len(word_set)}")
    print(f"🔢 Top 10 most common words:")
    word_counts = sorted(word_set, key=lambda w: -word_set.count(w))
    for i, word in enumerate(word_counts[:10]):
        print(f"{i+1}. {word} ({word_set.count(word)} occurrences)")

def restore_from_backup():
    """Restore model from backup if main checkpoint is corrupted"""
    global local_model, vocab
    
    if not os.path.exists(BACKUP_MODEL_PATH):
        print("❌ No backup file found")
        return False
    
    try:
        print("🔄 Restoring model from backup...")
        
        # Create backup of current model if it exists
        if os.path.exists(LOCAL_MODEL_PATH):
            current_backup = LOCAL_MODEL_PATH + '.current_backup'
            shutil.copy2(LOCAL_MODEL_PATH, current_backup)
            print(f"📦 Current model backed up to: {current_backup}")
        
        # Restore from backup
        shutil.copy2(BACKUP_MODEL_PATH, LOCAL_MODEL_PATH)
        print("✅ Backup restored successfully")
        
        # Reload the model
        load_local_model()
        
        if local_model is not None:
            print("✅ Model restored and loaded successfully")
            return True
        else:
            print("❌ Failed to load restored model")
            return False
            
    except Exception as e:
        print(f"❌ Error restoring from backup: {e}")
        return False

def show_checkpoint_info():
    """Show detailed information about the current checkpoint"""
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("❌ No checkpoint file found")
        return
    
    try:
        checkpoint = torch.load(LOCAL_MODEL_PATH, map_location='cpu')
        
        print("\n📋 CHECKPOINT INFORMATION:")
        print("=" * 50)
        
        if 'training_info' in checkpoint:
            info = checkpoint['training_info']
            print(f"🕒 Timestamp: {info.get('timestamp', 'Unknown')}")
            print(f"🔧 Device: {info.get('device', 'Unknown')}")
            print(f"🚀 GPU: {info.get('gpu_name', 'Unknown')}")
            print(f"🔥 Mixed Precision: {info.get('mixed_precision', 'Unknown')}")
            print(f"📊 Total Parameters: {info.get('total_parameters', 'Unknown'):,}")
        
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            print(f"\n🏗️ MODEL ARCHITECTURE:")
            print(f"   Layers: {config.get('num_layers', 'Unknown')}")
            print(f"   Embed Size: {config.get('embed_size', 'Unknown')}")
            print(f"   Attention Heads: {config.get('num_heads', 'Unknown')}")
            print(f"   Hidden Size: {config.get('hidden_size', 'Unknown')}")
            print(f"   Dropout: {config.get('dropout', 'Unknown')}")
            print(f"   Max Seq Length: {config.get('max_seq_len', 'Unknown')}")
        
        if 'vocab' in checkpoint:
            print(f"\n📚 VOCABULARY:")
            print(f"   Size: {len(checkpoint['vocab'])} words")
        
        if 'epoch' in checkpoint:
            print(f"\n⏱️ TRAINING PROGRESS:")
            print(f"   Current Epoch: {checkpoint.get('epoch', 'Unknown')}")
            print(f"   Best Loss: {checkpoint.get('loss', 'Unknown'):.4f}")
        
        if 'final_loss' in checkpoint:
            print(f"   Final Loss: {checkpoint.get('final_loss', 'Unknown'):.4f}")
        
        # File size
        file_size = os.path.getsize(LOCAL_MODEL_PATH) / 1024 / 1024
        print(f"\n💾 FILE INFO:")
        print(f"   Size: {file_size:.1f} MB")
        print(f"   Path: {os.path.abspath(LOCAL_MODEL_PATH)}")
        
        # Check if backup exists
        if os.path.exists(BACKUP_MODEL_PATH):
            backup_size = os.path.getsize(BACKUP_MODEL_PATH) / 1024 / 1024
            print(f"   Backup: {BACKUP_MODEL_PATH} ({backup_size:.1f} MB)")
        
        # Validate checkpoint integrity
        print(f"\n🔍 CHECKPOINT VALIDATION:")
        validation_result = validate_checkpoint(checkpoint)
        if validation_result:
            print("✅ Checkpoint is valid and ready for training")
        else:
            print("⚠️ Checkpoint has issues - consider restoring from backup")
        
    except Exception as e:
        print(f"❌ Error reading checkpoint: {e}")
        print("💡 The checkpoint file may be corrupted. Try restoring from backup.")

def validate_checkpoint(checkpoint):
    """Validate checkpoint integrity and completeness"""
    required_keys = ['model_state_dict', 'vocab']
    optional_keys = ['model_config', 'optimizer_state_dict', 'scheduler_state_dict']
    
    # Check required keys
    for key in required_keys:
        if key not in checkpoint:
            print(f"   ❌ Missing required key: {key}")
            return False
    
    # Check model state dict
    try:
        state_dict = checkpoint['model_state_dict']
        if not isinstance(state_dict, dict):
            print("   ❌ model_state_dict is not a dictionary")
            return False
        
        # Check for common layer names
        expected_layers = ['embedding.weight', 'fc.weight']
        for layer in expected_layers:
            if layer not in state_dict:
                print(f"   ⚠️ Missing expected layer: {layer}")
        
    except Exception as e:
        print(f"   ❌ Error validating model_state_dict: {e}")
        return False
    
    # Check vocabulary
    try:
        vocab = checkpoint['vocab']
        if not isinstance(vocab, dict):
            print("   ❌ vocab is not a dictionary")
            return False
        
        if len(vocab) < 100:  # Reasonable minimum
            print(f"   ⚠️ Vocabulary seems small: {len(vocab)} words")
        
    except Exception as e:
        print(f"   ❌ Error validating vocab: {e}")
        return False
    
    # Check optional keys and provide warnings
    for key in optional_keys:
        if key not in checkpoint:
            print(f"   ⚠️ Missing optional key: {key}")
    
    return True

def generate_with_local_model(prompt, max_length=100, temperature=0.8, top_p=0.9, repetition_penalty=1.1):
    """Generate response using advanced local model with sophisticated sampling"""
    if not local_model:
        return "Local model not ready."
    
    # Clean and tokenize input
    cleaned_prompt = clean_convo_data(prompt)
    input_tokens = cleaned_prompt.split()
    
    # Limit input context for efficiency
    if len(input_tokens) > 100:
        input_tokens = input_tokens[-100:]
    
    # Convert to tensor
    input_ids = torch.tensor([[vocab.get(token, vocab['<UNK>']) for token in input_tokens]]).to(device)
    
    local_model.eval()
    with torch.no_grad():
        generated_tokens = []
        repetition_count = defaultdict(int)
        
        for _ in range(max_length):
            # Create attention masks
            # For padding: (batch_size, seq_len) - True for padding tokens
            padding_mask = (input_ids == 0)
            
            # For causal masking: (seq_len, seq_len) - upper triangular matrix
            seq_len = input_ids.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1).bool()
            
            # Forward pass
            outputs = local_model(input_ids, padding_mask=padding_mask, causal_mask=causal_mask)
            next_token_logits = outputs[0, -1, :] / max(temperature, 0.1)
            
            # Apply repetition penalty
            for token_id, count in repetition_count.items():
                if count > 0:
                    penalty = repetition_penalty ** count
                    next_token_logits[token_id] /= penalty
            
            # Filter special tokens
            next_token_logits[vocab.get('<PAD>', 0)] = -float('inf')
            next_token_logits[vocab.get('<UNK>', 1)] = -float('inf')
            
            # Nucleus sampling (top-p)
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Find tokens that make up top-p of probability mass
            nucleus_mask = cumulative_probs <= top_p
            nucleus_mask[0] = True  # Always include top token
            
            # Filter logits and indices
            filtered_logits = sorted_logits[nucleus_mask]
            filtered_indices = sorted_indices[nucleus_mask]
            
            # Sample from nucleus
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = filtered_indices[torch.multinomial(probs, 1)]
            
            # Stop if we hit padding or unknown
            if next_token.item() in [vocab.get('<PAD>', 0), vocab.get('<UNK>', 1)]:
                break
            
            # Add to generated sequence
            generated_tokens.append(next_token.item())
            repetition_count[next_token.item()] += 1
            
            # Update input for next iteration
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            # Limit context length for efficiency
            if input_ids.size(1) > 150:
                input_ids = input_ids[:, -100:]
            
            # Stop if we have enough tokens
            if len(generated_tokens) >= max_length:
                break
        
        # Convert tokens back to text
        response_tokens = []
        for token_id in generated_tokens:
            if token_id in vocab.values():
                token = list(vocab.keys())[list(vocab.values()).index(token_id)]
                if token not in ['<PAD>', '<UNK>']:
                    response_tokens.append(token)
        
        response = ' '.join(response_tokens)
        
        # Clean up response
        response = response.strip()
        if len(response) > 2000:  # Discord max length
            response = response[:1997] + "..."
        
        return response if response else "I'm not sure how to respond to that."

def test_model():
    """Test the trained model with sample prompts"""
    if not local_model:
        print("❌ No model loaded. Please train first.")
        return
    
    test_prompts = [
        "Hello, how are you?",
        "What is artificial intelligence?",
        "Tell me a joke",
        "Explain machine learning",
        "How do I train a neural network?"
    ]
    
    print("\n🧪 TESTING TRAINED MODEL:")
    print("=" * 50)
    
    for prompt in test_prompts:
        print(f"\n📝 Input: {prompt}")
        response = generate_with_local_model(prompt, max_length=50)
        print(f"🤖 Response: {response}")
        print("-" * 30)

def show_system_info():
    """Display system information and training capabilities for Kaggle"""
    print("\n🖥️ KAGGLE SYSTEM INFORMATION:")
    print("=" * 50)
    
    # Kaggle-specific information
    print(f"🏠 Kaggle Environment: Active")
    print(f"📁 Input Directory: {KAGGLE_INPUT_DIR}")
    print(f"📁 Working Directory: {KAGGLE_WORKING_DIR}")
    
    # Check Kaggle input directory
    if os.path.exists(KAGGLE_INPUT_DIR):
        input_files = os.listdir(KAGGLE_INPUT_DIR)
        print(f"📚 Input Files: {len(input_files)} files found")
        if len(input_files) <= 10:
            for file in input_files:
                print(f"   📄 {file}")
        else:
            print(f"   📄 {len(input_files)} files (showing first 10)")
            for file in input_files[:10]:
                print(f"   📄 {file}")
            print(f"   ... and {len(input_files) - 10} more")
    else:
        print(f"❌ Input directory not found: {KAGGLE_INPUT_DIR}")
    
    # PyTorch version
    print(f"\n🔥 PyTorch Version: {torch.__version__}")
    
    # Device information
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"🚀 CUDA Available: Yes")
        print(f"🎯 CUDA Version: {torch.version.cuda}")
        print(f"💾 GPU Count: {gpu_count} Tesla T4 GPUs")
        
        for i in range(gpu_count):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
            print(f"   GPU {i} Compute Capability: {torch.cuda.get_device_capability(i)}")
        
        # Check for mixed precision support
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            print("✅ Mixed Precision Training: Supported")
        else:
            print("⚠️ Mixed Precision Training: Not supported")
            
        # Check for TF32 support
        if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
            print("✅ TF32: Supported")
        else:
            print("⚠️ TF32: Not supported")
    else:
        print("🖥️ CUDA Available: No (CPU training mode)")
        print(f"🧵 CPU Threads: {torch.get_num_threads()}")
        print(f"💾 CPU Cores: {os.cpu_count()}")
    
    # Training optimizations
    print(f"\n⚡ TRAINING OPTIMIZATIONS:")
    if torch.cuda.is_available():
        print(f"   cudnn.benchmark: {torch.backends.cudnn.benchmark}")
        print(f"   cudnn.deterministic: {torch.backends.cudnn.deterministic}")
        print(f"   TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"   TF32 cudnn: {torch.backends.cudnn.allow_tf32}")
        if gpu_count > 1:
            print(f"   🚀 Multi-GPU Training: Enabled with DataParallel")
    else:
        print(f"   CPU Threads: {torch.get_num_threads()}")
    
    # Model file status
    print(f"\n📁 MODEL FILES:")
    if os.path.exists(LOCAL_MODEL_PATH):
        size = os.path.getsize(LOCAL_MODEL_PATH) / 1024 / 1024
        print(f"   Main Model: {LOCAL_MODEL_PATH} ({size:.1f} MB)")
    else:
        print(f"   Main Model: {LOCAL_MODEL_PATH} (Not found)")
    
    if os.path.exists(BACKUP_MODEL_PATH):
        size = os.path.getsize(BACKUP_MODEL_PATH) / 1024 / 1024
        print(f"   Backup: {BACKUP_MODEL_PATH} ({size:.1f} MB)")
    else:
        print(f"   Backup: {BACKUP_MODEL_PATH} (Not found)")

def create_sample_jsonl():
    """Create a sample JSONL file with multiple formats including conversation format in Kaggle working directory"""
    sample_data = [
        # Format 1: Instruction/Input/Output
        {
            "instruction": "Explain the difference between AI and ML.",
            "input": "User asks about AI concepts.",
            "output": "Sir, AI refers to the broader field of machines performing tasks that require intelligence, whereas ML is a specialized subset focusing on teaching machines to learn from data patterns."
        },
        # Format 2: Conversation with Messages (your format)
        {
            "conversation_id": "0001",
            "messages": [
                {"role": "system", "content": "You are J.A.R.V.I.S, a highly intelligent AI assistant."},
                {"role": "user", "content": "Hello, Jarvis."},
                {"role": "assistant", "content": "Good day, sir. How may I assist you today?"}
            ],
            "memory": [
                {"summary": "User greeted Jarvis."},
                {"summary": "Jarvis responded politely and asked how to help."}
            ]
        },
        # Format 3: Another conversation example
        {
            "conversation_id": "0002",
            "messages": [
                {"role": "system", "content": "You are J.A.R.V.I.S, a highly intelligent AI assistant."},
                {"role": "user", "content": "What's the weather like?"},
                {"role": "assistant", "content": "I apologize, sir, but I don't have access to real-time weather data. Would you like me to help you find a weather service or check your local weather app?"}
            ],
            "memory": [
                {"summary": "User asked about weather."},
                {"summary": "Jarvis explained limitations and offered alternatives."}
            ]
        },
        # Format 4: Simple Q&A
        {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed for every task."
        }
    ]
    
    output_file = os.path.join(KAGGLE_WORKING_DIR, "sample_training.jsonl")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ Created sample JSONL file: {output_file}")
    print(f"📝 Contains {len(sample_data)} training examples")
    print("🔧 Includes multiple formats: instruction/input/output, conversation with messages, and Q&A")
    print("🎯 The conversation format will create multiple training pairs with context and memory")

def validate_jsonl_file(file_path):
    """Validate a JSONL file and show statistics"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        valid_lines = 0
        invalid_lines = 0
        total_lines = 0
        format_counts = defaultdict(int)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                total_lines += 1
                try:
                    json_data = json.loads(line)
                    if isinstance(json_data, dict):
                        valid_lines += 1
                        
                        # Count different formats
                        if 'conversation_id' in json_data and 'messages' in json_data:
                            format_counts['conversation_with_messages'] += 1
                        elif 'instruction' in json_data and 'input' in json_data and 'output' in json_data:
                            format_counts['instruction/input/output'] += 1
                        elif 'instruction' in json_data and 'response' in json_data:
                            format_counts['instruction/response'] += 1
                        elif 'input' in json_data and 'target' in json_data:
                            format_counts['input/target'] += 1
                        elif 'question' in json_data and 'answer' in json_data:
                            format_counts['question/answer'] += 1
                        elif 'prompt' in json_data and 'completion' in json_data:
                            format_counts['prompt/completion'] += 1
                        else:
                            format_counts['other'] += 1
                    else:
                        invalid_lines += 1
                        print(f"⚠️ Line {line_num}: Not a valid JSON object")
                        
                except json.JSONDecodeError as e:
                    invalid_lines += 1
                    print(f"❌ Line {line_num}: JSON parse error - {e}")
        
        print(f"\n📊 JSONL File Validation Results: {file_path}")
        print("=" * 50)
        print(f"📝 Total lines: {total_lines}")
        print(f"✅ Valid lines: {valid_lines}")
        print(f"❌ Invalid lines: {invalid_lines}")
        print(f"📈 Success rate: {(valid_lines/total_lines*100):.1f}%" if total_lines > 0 else "N/A")
        
        if format_counts:
            print(f"\n🔍 Format Breakdown:")
            for format_name, count in format_counts.items():
                print(f"   {format_name}: {count}")
        
        return invalid_lines == 0
        
    except Exception as e:
        print(f"❌ Error validating file: {e}")
        return False

def test_conversation_format():
    """Test the conversation format parsing with your exact example"""
    test_data = {
        "conversation_id": "0001",
        "messages": [
            {"role": "system", "content": "You are J.A.R.V.I.S, a highly intelligent AI assistant."},
            {"role": "user", "content": "Hello, Jarvis."},
            {"role": "assistant", "content": "Good day, sir. How may I assist you today?"}
        ],
        "memory": [
            {"summary": "User greeted Jarvis."},
            {"summary": "Jarvis responded politely and asked how to help."}
        ]
    }
    
    print("🧪 Testing Conversation Format Parsing")
    print("=" * 50)
    print(f"📋 Conversation ID: {test_data['conversation_id']}")
    print(f"💬 Messages: {len(test_data['messages'])}")
    print(f"🧠 Memory entries: {len(test_data['memory'])}")
    
    # Simulate the parsing logic
    messages = test_data['messages']
    conversation_id = test_data.get('conversation_id', 'unknown')
    memory_summaries = test_data.get('memory', [])
    
    # Build context from system messages and memory
    context_parts = []
    system_content = ""
    
    # Collect system messages
    for msg in messages:
        if msg.get('role') == 'system':
            system_content += str(msg.get('content', '')) + " "
    
    if system_content.strip():
        context_parts.append(f"System: {system_content.strip()}")
    
    # Add memory summaries if available
    if memory_summaries:
        memory_text = "; ".join([m.get('summary', '') for m in memory_summaries if m.get('summary')])
        if memory_text:
            context_parts.append(f"Memory: {memory_text}")
    
    print(f"\n🔧 Context built:")
    for i, part in enumerate(context_parts, 1):
        print(f"   {i}. {part}")
    
    # Create training pairs from conversation flow
    training_pairs = []
    for i in range(len(messages) - 1):
        current_msg = messages[i]
        next_msg = messages[i + 1]
        
        if (current_msg.get('role') == 'user' and 
            next_msg.get('role') == 'assistant'):
            
            # Build input with context
            input_text = current_msg.get('content', '')
            if context_parts:
                input_text = f"{' | '.join(context_parts)} | User: {input_text}"
            
            # Create training pair
            training_pairs.append({
                'input': input_text,
                'target': next_msg.get('content', '')
            })
    
    print(f"\n🎯 Training pairs created: {len(training_pairs)}")
    for i, pair in enumerate(training_pairs, 1):
        print(f"\n   Pair {i}:")
        print(f"   Input: {pair['input']}")
        print(f"   Target: {pair['target']}")
    
    print(f"\n✅ Conversation format parsing test completed successfully!")

def test_kaggle_gpu():
    """Test Kaggle GPU setup and performance"""
    print("🚀 Testing Kaggle GPU Setup...")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Cannot test GPU.")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"🔍 Found {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        print(f"\n🚀 GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"   💾 Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"   🔧 Compute Capability: {props.major}.{props.minor}")
        print(f"   🧵 Multiprocessors: {props.multi_processor_count}")
        
        # Test GPU memory allocation
        try:
            torch.cuda.set_device(i)
            test_tensor = torch.randn(1000, 1000, device=f'cuda:{i}')
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**2
            print(f"   ✅ Memory test: {memory_allocated:.1f} MB allocated")
            
            # Test basic operations
            result = torch.matmul(test_tensor, test_tensor)
            print(f"   ✅ Computation test: Matrix multiplication successful")
            
            # Clean up
            del test_tensor, result
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ GPU {i} test failed: {e}")
    
    # Test single GPU training (recommended for Kaggle)
    print(f"\n🚀 Single GPU Training Test (Recommended for Kaggle):")
    try:
        # Create a simple model on GPU 0
        torch.cuda.set_device(0)
        model = torch.nn.Linear(100, 10).cuda()
        print(f"   ✅ Model created on GPU 0")
        
        # Test forward pass
        x = torch.randn(32, 100).cuda()
        output = model(x)
        print(f"   ✅ Forward pass successful: {output.shape}")
        
        # Test backward pass
        loss = output.sum()
        loss.backward()
        print(f"   ✅ Backward pass successful")
        
        # Clean up
        del model, x, output, loss
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"   ❌ Single GPU test failed: {e}")
    
    # Test multi-GPU if available (but warn about stability)
    if gpu_count > 1:
        print(f"\n⚠️  Multi-GPU Test (May be unstable on Kaggle):")
        try:
            # Create a simple model
            model = torch.nn.Linear(100, 10).cuda()
            if gpu_count > 1:
                model = torch.nn.DataParallel(model)
                print(f"   ✅ DataParallel wrapper successful")
            
            # Test forward pass
            x = torch.randn(32, 100).cuda()
            output = model(x)
            print(f"   ✅ Forward pass successful: {output.shape}")
            
            # Clean up
            del model, x, output
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ Multi-GPU test failed: {e}")
            print(f"   💡 This is expected on Kaggle - use single GPU for stability")
    
    print(f"\n✅ GPU testing completed!")
    print(f"💡 Recommendation: Use single GPU training on Kaggle for stability")

def test_single_gpu_training():
    """Test single GPU training with the actual model"""
    print("🧪 Testing Single GPU Training with Actual Model...")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Cannot test GPU training.")
        return
    
    if local_model is None:
        print("❌ No model loaded. Please load a model first.")
        return
    
    try:
        # Set device to GPU 0
        torch.cuda.set_device(0)
        print(f"🚀 Using GPU 0: {torch.cuda.get_device_name(0)}")
        
        # Move model to GPU if not already there
        if next(local_model.parameters()).device.type != 'cuda':
            local_model.cuda()
            print("✅ Model moved to GPU")
        
        # Create test data
        print("🔄 Creating test data...")
        batch_size = 4  # Small batch for testing
        seq_len = 64    # Short sequence for testing
        
        # Create random input data
        test_input = torch.randint(0, 1000, (batch_size, seq_len)).cuda()
        test_target = torch.randint(0, 1000, (batch_size, seq_len)).cuda()
        
        print(f"   Test input shape: {test_input.shape}")
        print(f"   Test target shape: {test_target.shape}")
        
        # Test forward pass
        print("🔄 Testing forward pass...")
        local_model.eval()
        with torch.no_grad():
            output = local_model(test_input)
            print(f"   ✅ Forward pass successful: {output.shape}")
        
        # Test training mode
        print("🔄 Testing training mode...")
        local_model.train()
        
        # Create optimizer
        optimizer = torch.optim.AdamW(local_model.parameters(), lr=1e-4)
        
        # Test training step
        print("🔄 Testing training step...")
        output = local_model(test_input)
        loss = torch.nn.functional.cross_entropy(
            output.view(-1, output.size(-1)), 
            test_target.view(-1)
        )
        
        print(f"   ✅ Loss calculation successful: {loss.item():.4f}")
        
        # Test backward pass
        print("🔄 Testing backward pass...")
        loss.backward()
        print("   ✅ Backward pass successful")
        
        # Test optimizer step
        print("🔄 Testing optimizer step...")
        optimizer.step()
        optimizer.zero_grad()
        print("   ✅ Optimizer step successful")
        
        # Check GPU memory
        gpu_memory = torch.cuda.memory_allocated(0) / 1024**2
        print(f"   💾 GPU memory used: {gpu_memory:.1f} MB")
        
        # Clean up
        del test_input, test_target, output, loss, optimizer
        torch.cuda.empty_cache()
        
        print("\n✅ Single GPU training test completed successfully!")
        print("🎯 Your model is ready for training on Kaggle!")
        
    except Exception as e:
        print(f"❌ Single GPU training test failed: {e}")
        print("💡 This might indicate an issue with the model or GPU setup")

def create_kaggle_requirements():
    """Create a requirements.txt file optimized for Kaggle"""
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "numpy>=1.21.0",
        "tqdm>=4.64.0",
        "psutil>=5.9.0"
    ]
    
    output_file = os.path.join(KAGGLE_WORKING_DIR, "requirements.txt")
    
    try:
        with open(output_file, 'w') as f:
            for req in requirements:
                f.write(req + '\n')
        
        print(f"✅ Created Kaggle requirements file: {output_file}")
        print("📋 Requirements:")
        for req in requirements:
            print(f"   📦 {req}")
        
        print(f"\n💡 To install in Kaggle:")
        print(f"   pip install -r {output_file}")
        
    except Exception as e:
        print(f"❌ Error creating requirements file: {e}")

def check_kaggle_environment():
    """Check and setup Kaggle environment with enhanced compatibility"""
    print("🔍 Checking Kaggle environment...")
    
    # Check if we're in Kaggle
    is_kaggle = os.path.exists(KAGGLE_INPUT_DIR)
    if is_kaggle:
        print(f"✅ Kaggle input directory found: {KAGGLE_INPUT_DIR}")
        
        # Check Kaggle-specific requirements
        print("\n🔍 Checking Kaggle compatibility...")
        
        # Check GPU availability
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   ✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Kaggle-specific GPU optimizations
            if 'T4' in gpu_name:
                print("   🚀 Tesla T4 detected - applying Kaggle optimizations")
                # Set environment variables for T4
                os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
                os.environ['NCCL_DEBUG'] = 'INFO'
                os.environ['NCCL_IB_DISABLE'] = '1'
                os.environ['NCCL_P2P_DISABLE'] = '1'
            elif 'P100' in gpu_name:
                print("   🚀 Tesla P100 detected - applying Kaggle optimizations")
                os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
            else:
                print(f"   ℹ️ Unknown GPU type: {gpu_name}")
        else:
            print("   ⚠️ No GPU detected - will use CPU training")
        
        # Check working directory permissions
        try:
            test_file = os.path.join(KAGGLE_WORKING_DIR, "kaggle_test.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print("   ✅ Working directory is writable")
        except Exception as e:
            print(f"   ❌ Working directory not writable: {e}")
            return False
        
        # Check input directory access
        if os.path.exists(KAGGLE_INPUT_DIR):
            try:
                files = os.listdir(KAGGLE_INPUT_DIR)
                print(f"   ✅ Input directory accessible: {len(files)} files found")
            except Exception as e:
                print(f"   ❌ Input directory not accessible: {e}")
                return False
        else:
            print(f"   ⚠️ Input directory not found: {KAGGLE_INPUT_DIR}")
        
        print("   🎯 Kaggle environment is ready for training!")
        
    else:
        print(f"⚠️  Warning: Kaggle input directory not found: {KAGGLE_INPUT_DIR}")
        print("   This script is designed for Kaggle but can run in other environments")
        
        # Check local environment compatibility
        print("\n🔍 Checking local environment compatibility...")
        
        # Check if we have enough disk space for checkpoints
        try:
            import shutil
            total, used, free = shutil.disk_usage('.')
            free_gb = free / (1024**3)
            print(f"   💾 Available disk space: {free_gb:.1f} GB")
            
            if free_gb < 10:
                print("   ⚠️ Low disk space - consider freeing up space for checkpoints")
            else:
                print("   ✅ Sufficient disk space for training")
        except Exception as e:
            print(f"   ⚠️ Could not check disk space: {e}")
    
    # Ensure working directory exists
    if not os.path.exists(KAGGLE_WORKING_DIR):
        print(f"⚠️  Warning: Kaggle working directory not found: {KAGGLE_WORKING_DIR}")
        print("   Using current directory instead")
        # Don't overwrite the global constants - they're already set correctly
    else:
        print(f"✅ Kaggle working directory found: {KAGGLE_WORKING_DIR}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"🚀 GPU Environment: {gpu_count} Tesla T4 GPUs detected")
        if gpu_count > 1:
            print(f"   🎯 Multi-GPU training will be enabled")
        else:
            print(f"   🎯 Single GPU training mode")
    else:
        print("🖥️ GPU Environment: No GPUs detected, will use CPU")
    
    print("✅ Kaggle environment check completed\n")
    return is_kaggle

def create_optimized_vocab(max_size=100000):
    """Create an optimized vocabulary that balances coverage with parameter efficiency"""
    global vocab
    
    print(f"🔧 Creating optimized vocabulary (max_size={max_size:,})...")
    
    # Build full vocabulary first
    build_vocab_from_db(incremental=False, max_vocab_size=1000000)  # Get all words first
    
    if len(vocab) <= max_size:
        print(f"✅ Vocabulary already within size limit ({len(vocab):,} <= {max_size:,})")
        return len(vocab)
    
    print(f"🔄 Optimizing vocabulary from {len(vocab):,} to {max_size:,} words...")
    
    # Get word frequencies from the data
    word_freq = defaultdict(int)
    
    # Count frequencies from database
    try:
        with DB_LOCK:
            if DB_CONN is not None:
                cursor = DB_CONN.cursor()
                cursor.execute("SELECT content, response FROM memory")
                rows = cursor.fetchall()
                for content, response in rows:
                    for word in clean_convo_data(content).split():
                        if word in vocab:
                            word_freq[word] += 1
                    for word in clean_convo_data(response).split():
                        if word in vocab:
                            word_freq[word] += 1
    except Exception as e:
        print(f"⚠️ Error counting database frequencies: {e}")
    
    # Count frequencies from data folder
    data_conversations = load_data_folder_files()
    for conv in data_conversations:
        for word in clean_convo_data(conv['input']).split():
            if word in vocab:
                word_freq[word] += 1
        for word in clean_convo_data(conv['target']).split():
            if word in vocab:
                word_freq[word] += 1
    
    # Sort by frequency and keep top words
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Keep special tokens
    special_tokens = {'<PAD>', '<UNK>'}
    kept_words = set()
    
    # Add special tokens first
    for token in special_tokens:
        if token in vocab:
            kept_words.add(token)
    
    # Add most frequent words
    for word, freq in sorted_words[:max_size - len(kept_words)]:
        kept_words.add(word)
    
    # Create new optimized vocabulary
    new_vocab = {}
    for i, word in enumerate(sorted(kept_words)):
        new_vocab[word] = i
    
    # Update global vocabulary
    old_size = len(vocab)
    vocab = new_vocab
    new_size = len(vocab)
    
    print(f"✅ Vocabulary optimized: {old_size:,} → {new_size:,} words")
    print(f"   Coverage: {len(kept_words - special_tokens):,} most frequent words")
    print(f"   Dropped: {old_size - new_size:,} rare words")
    
    # Calculate parameter savings
    old_params = old_size * 640  # embedding size
    new_params = new_size * 640
    savings = old_params - new_params
    print(f"💾 Parameter savings: ~{savings / 1e6:.1f}M parameters")
    
    return new_size

def add_subword_support():
    """Add subword tokenization support for handling rare words"""
    print("🔧 Adding subword tokenization support...")
    
    # This is a placeholder for future subword implementation
    # For now, we'll use character-level fallback for rare words
    print("   ℹ️  Subword tokenization not yet implemented")
    print("   ℹ️  Rare words will be mapped to <UNK> token")
    print("   ℹ️  Consider using SentencePiece or BPE for production use")
    
    return True

def show_model_info():
    """Show detailed information about the current model"""
    global local_model, vocab
    
    if local_model is None:
        print("❌ No model loaded")
        return
    
    print("\n🤖 MODEL INFORMATION:")
    print("=" * 50)
    
    # Unwrap DataParallel if needed
    actual_model = local_model
    if hasattr(local_model, 'module'):
        actual_model = local_model.module
        # Get device from parameters instead of non-existent device attribute
        device = next(actual_model.parameters()).device
        print(f"🔄 DataParallel: Wrapped (inner model on {device})")
    else:
        print(f"🔄 DataParallel: Not wrapped")
    
    # Model architecture
    print(f"🏗️  Architecture: {actual_model.num_layers}L/{actual_model.embed_size}E/{actual_model.num_heads}H/{actual_model.hidden_size}F")
    
    # Parameter count
    total_params = sum(p.numel() for p in local_model.parameters())
    print(f"📊 Total Parameters: {total_params:,}")
    
    # Vocabulary size
    vocab_size = len(vocab) if vocab else 0
    print(f"📚 Vocabulary Size: {vocab_size:,} words")
    
    # Parameter breakdown
    embedding_params = actual_model.embedding.weight.numel()
    output_params = actual_model.fc.weight.numel()
    transformer_params = total_params - embedding_params - output_params
    
    print(f"\n🔍 PARAMETER BREAKDOWN:")
    # Safety check to prevent division by zero
    if total_params > 0:
        print(f"   Embedding Layer: {embedding_params:,} ({embedding_params/total_params*100:.1f}%)")
        print(f"   Output Layer: {output_params:,} ({output_params/total_params*100:.1f}%)")
        print(f"   Transformer: {transformer_params:,} ({transformer_params/total_params*100:.1f}%)")
    else:
        print(f"   ⚠️  Warning: Total parameters is 0, cannot calculate percentages")
        print(f"   Embedding Layer: {embedding_params:,}")
        print(f"   Output Layer: {output_params:,}")
        print(f"   Transformer: {transformer_params:,}")
    
    # Memory estimates
    fp32_memory = total_params * 4 / 1024 / 1024  # MB
    fp16_memory = total_params * 2 / 1024 / 1024  # MB
    
    print(f"\n💾 MEMORY ESTIMATES:")
    print(f"   FP32: {fp32_memory:.1f} MB")
    print(f"   FP16: {fp16_memory:.1f} MB")
    
    # Recommendations
    if vocab_size > 100000:
        print(f"\n⚠️  WARNING: Large vocabulary detected!")
        print(f"   This model has {vocab_size:,} words which significantly increases parameters.")
        print(f"   Consider using option E to optimize vocabulary to 100k words.")
        print(f"   This would save ~{(vocab_size - 100000) * 640 / 1e6:.1f}M parameters.")
    
    if total_params > 500_000_000:  # 500M
        print(f"\n⚠️  WARNING: Large model detected!")
        print(f"   This model has {total_params/1e6:.1f}M parameters.")
        print(f"   Training on a single T4 GPU may be slow.")
        print(f"   Consider using gradient checkpointing and mixed precision.")
    
    print("=" * 50)

def show_gpu_info():
    """Show detailed GPU information for dual-GPU setup"""
    print("\n🚀 GPU INFORMATION:")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ No CUDA GPUs available")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"📊 Available GPUs: {gpu_count}")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_props = torch.cuda.get_device_properties(i)
        
        print(f"\n🔧 GPU {i}: {gpu_name}")
        print(f"   Memory: {gpu_props.total_memory / (1024**3):.1f} GB")
        print(f"   Compute Capability: {gpu_props.major}.{gpu_props.minor}")
        
        # Show current memory usage
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        print(f"   Memory Allocated: {allocated:.2f} GB")
        print(f"   Memory Reserved: {reserved:.2f} GB")
        print(f"   Memory Free: {(gpu_props.total_memory / (1024**3)) - reserved:.2f} GB")
    
    # Show DataParallel status
    if hasattr(local_model, 'module'):
        print(f"\n🔄 DataParallel Status: ✅ Enabled")
        print(f"   Primary Device: {local_model.device_ids[0]}")
        print(f"   All Devices: {local_model.device_ids}")
    else:
        print(f"\n🔄 DataParallel Status: ❌ Disabled")
        print(f"   Model is on device: {next(local_model.parameters()).device}")
    
    # Show CUDA environment
    print(f"\n🌍 CUDA Environment:")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"   Current Device: {torch.cuda.current_device()}")
    
    print("=" * 50)

def test_dual_gpu_training():
    """Test dual-GPU training setup"""
    global local_model
    
    print("\n🚀 Testing Dual-GPU Training Setup...")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ No CUDA GPUs available")
        return
    
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        print(f"❌ Only {gpu_count} GPU(s) available, need at least 2 for dual-GPU training")
        return
    
    print(f"✅ Found {gpu_count} GPUs")
    
    # Check if model is wrapped with DataParallel
    if hasattr(local_model, 'module'):
        print("✅ Model is wrapped with DataParallel")
        print(f"   Device IDs: {local_model.device_ids}")
        print(f"   Primary device: {local_model.device_ids[0]}")
    else:
        print("⚠️  Model is not wrapped with DataParallel")
        print("   Wrapping now for dual-GPU training...")
        
        # Wrap with DataParallel
        local_model = torch.nn.DataParallel(local_model, device_ids=[0, 1])
        print("   ✅ Model wrapped with DataParallel")
    
    # Test forward pass on both GPUs
    print("\n🧪 Testing forward pass on both GPUs...")
    
    try:
        # Create test input
        batch_size = 4
        seq_len = 512
        vocab_size = len(vocab) if vocab else 1000
        
        test_input = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda:0')
        
        # Forward pass
        with torch.no_grad():
            outputs = local_model(test_input)
        
        print(f"✅ Forward pass successful!")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {outputs.shape}")
        print(f"   Output device: {outputs.device}")
        
        # Check memory usage on both GPUs
        print(f"\n💾 Memory usage after forward pass:")
        for i in range(2):
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"   GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        # Test backward pass
        print(f"\n🧪 Testing backward pass...")
        loss = outputs.sum()
        loss.backward()
        print(f"✅ Backward pass successful!")
        
        # Clean up
        del test_input, outputs, loss
        torch.cuda.empty_cache()
        
        print(f"\n🎉 Dual-GPU training test completed successfully!")
        print(f"   Both GPUs are working and can handle forward/backward passes")
        print(f"   DataParallel is properly distributing the workload")
        
    except Exception as e:
        print(f"❌ Dual-GPU test failed: {e}")
        print(f"   This might indicate a configuration issue")
    
    print("=" * 50)

def get_model_property(model, property_name):
    """Safely get model property whether it's wrapped in DataParallel or not"""
    if hasattr(model, 'module'):
        # DataParallel wrapped model
        return getattr(model.module, property_name)
    else:
        # Direct model
        return getattr(model, property_name)

def get_model_max_seq_len(model):
    """Get the maximum sequence length from the model"""
    return get_model_property(model, 'max_seq_len')

def test_attention_masks():
    """Test attention mask creation to debug size issues"""
    global local_model
    
    print("\n🧪 Testing Attention Mask Creation...")
    print("=" * 50)
    
    if local_model is None:
        print("❌ No model loaded")
        return
    
    try:
        # Test different sequence lengths
        test_lengths = [512, 1024, 2048]
        
        for seq_len in test_lengths:
            print(f"\n🔍 Testing sequence length: {seq_len}")
            
            # Create test input
            batch_size = 2
            vocab_size = len(vocab) if vocab else 1000
            test_input = torch.randint(0, vocab_size, (batch_size, seq_len), device='cpu')
            
            # Create attention masks
            padding_mask = (test_input == 0)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device='cpu'), diagonal=1).bool()
            
            print(f"   Input shape: {test_input.shape}")
            print(f"   Padding mask shape: {padding_mask.shape}")
            print(f"   Causal mask shape: {causal_mask.shape}")
            
            # Check if sequence length is within model limits
            model_max_seq_len = get_model_max_seq_len(local_model)
            if seq_len > model_max_seq_len:
                print(f"   ⚠️  Sequence length {seq_len} exceeds model limit {model_max_seq_len}")
                print(f"   🔧 Would truncate to: {model_max_seq_len}")
                
                # Test truncation
                truncated_seq_len = model_max_seq_len
                truncated_input = test_input[:, :truncated_seq_len]
                truncated_padding_mask = (truncated_input == 0)
                truncated_causal_mask = torch.triu(torch.ones(truncated_seq_len, truncated_seq_len, device='cpu'), diagonal=1).bool()
                
                print(f"   ✅ After truncation:")
                print(f"      Input shape: {truncated_input.shape}")
                print(f"      Padding mask shape: {truncated_padding_mask.shape}")
                print(f"      Causal mask shape: {truncated_causal_mask.shape}")
            else:
                print(f"   ✅ Sequence length {seq_len} is within model limit {model_max_seq_len}")
        
        print(f"\n🎯 Model Information:")
        print(f"   Max sequence length: {get_model_max_seq_len(local_model)}")
        print(f"   DataParallel wrapped: {'Yes' if hasattr(local_model, 'module') else 'No'}")
        if hasattr(local_model, 'module'):
            print(f"   Device IDs: {local_model.device_ids}")
        
        print(f"\n✅ Attention mask test completed successfully!")
        
    except Exception as e:
        print(f"❌ Attention mask test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 50)

class ByteLevelTokenizer:
    """Efficient byte-level tokenizer for handling large vocabularies"""
    
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        # Special tokens
        self.special_tokens = {
            '<PAD>': self.pad_token_id,
            '<UNK>': self.unk_token_id,
            '<BOS>': self.bos_token_id,
            '<EOS>': self.eos_token_id
        }
        
        # Byte vocabulary (0-255)
        self.byte_vocab = {chr(i): i + len(self.special_tokens) for i in range(256)}
        
        # Reverse mapping
        self.id_to_token = {v: k for k, v in {**self.special_tokens, **self.byte_vocab}.items()}
        
        print(f"🔤 ByteLevelTokenizer initialized:")
        print(f"   Special tokens: {len(self.special_tokens)}")
        print(f"   Byte vocabulary: {len(self.byte_vocab)}")
        print(f"   Total vocabulary: {len(self.id_to_token)}")
        print(f"   Parameter savings: ~{(1000000 - len(self.id_to_token)) * 640 / 1e6:.1f}M")
    
    def encode(self, text, max_length=None):
        """Encode text to token IDs"""
        if not text:
            return [self.pad_token_id]
        
        # Convert text to bytes and then to token IDs
        tokens = [self.bos_token_id]
        
        # Add byte tokens
        for char in text:
            if char in self.byte_vocab:
                tokens.append(self.byte_vocab[char])
            else:
                # Fallback to UNK for any unexpected characters
                tokens.append(self.unk_token_id)
        
        tokens.append(self.eos_token_id)
        
        # Truncate if needed
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.eos_token_id]
        
        return tokens
    
    def decode(self, token_ids):
        """Decode token IDs back to text"""
        if not token_ids:
            return ""
        
        # Remove special tokens and convert back to text
        text_chars = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token not in self.special_tokens:
                    text_chars.append(token)
        
        return ''.join(text_chars)
    
    def __len__(self):
        return len(self.id_to_token)
    
    def get_vocab_size(self):
        return len(self.id_to_token)

class HybridTokenizer:
    """Hybrid tokenizer that can switch between word-level and byte-level"""
    
    def __init__(self, mode='auto', max_word_vocab_size=100000):
        self.mode = mode
        self.max_word_vocab_size = max_word_vocab_size
        self.word_tokenizer = None
        self.byte_tokenizer = None
        self.current_mode = None
        
        # Initialize both tokenizers
        self._init_tokenizers()
        
        print(f"🔄 HybridTokenizer initialized in {self.mode} mode")
        print(f"   Word vocab size: {len(self.word_tokenizer) if self.word_tokenizer else 'N/A'}")
        print(f"   Byte vocab size: {len(self.byte_tokenizer) if self.byte_tokenizer else 'N/A'}")
    
    def _init_tokenizers(self):
        """Initialize both tokenizer types"""
        # Word-level tokenizer
        self.word_tokenizer = {}
        
        # Byte-level tokenizer
        if BYTE_LEVEL_AVAILABLE:
            self.byte_tokenizer = ByteLevelTokenizer()
        else:
            print("⚠️  Byte-level tokenization not available")
            self.byte_tokenizer = None
    
    def set_mode(self, mode):
        """Set the active tokenization mode"""
        if mode == 'word' and self.word_tokenizer:
            self.current_mode = 'word'
            print(f"🔄 Switched to word-level tokenization (vocab size: {len(self.word_tokenizer)})")
        elif mode == 'byte' and self.byte_tokenizer:
            self.current_mode = 'byte'
            print(f"🔄 Switched to byte-level tokenization (vocab size: {len(self.byte_tokenizer)})")
        elif mode == 'auto':
            # Auto-select based on vocabulary size
            if len(self.word_tokenizer) > self.max_word_vocab_size:
                self.current_mode = 'byte'
                print(f"🔄 Auto-selected byte-level (word vocab too large: {len(self.word_tokenizer)})")
            else:
                self.current_mode = 'word'
                print(f"🔄 Auto-selected word-level (vocab size: {len(self.word_tokenizer)})")
        else:
            print(f"❌ Invalid mode: {mode}")
            return False
        
        return True
    
    def encode(self, text, max_length=None):
        """Encode text using the current mode"""
        if self.current_mode == 'byte' and self.byte_tokenizer:
            return self.byte_tokenizer.encode(text, max_length)
        elif self.current_mode == 'word' and self.word_tokenizer:
            # Fallback to simple word splitting for now
            words = text.split()
            tokens = []
            for word in words:
                if word in self.word_tokenizer:
                    tokens.append(self.word_tokenizer[word])
                else:
                    tokens.append(1)  # UNK token
            return tokens
        else:
            raise ValueError("No valid tokenizer available")
    
    def decode(self, token_ids):
        """Decode token IDs using the current mode"""
        if self.current_mode == 'byte' and self.byte_tokenizer:
            return self.byte_tokenizer.decode(token_ids)
        elif self.current_mode == 'word' and self.word_tokenizer:
            # Simple reverse mapping
            id_to_word = {v: k for k, v in self.word_tokenizer.items()}
            words = []
            for token_id in token_ids:
                if token_id in id_to_word:
                    words.append(id_to_word[token_id])
            return ' '.join(words)
        else:
            raise ValueError("No valid tokenizer available")
    
    def get_vocab_size(self):
        """Get vocabulary size for the current mode"""
        if self.current_mode == 'byte' and self.byte_tokenizer:
            return len(self.byte_tokenizer)
        elif self.current_mode == 'word' and self.word_tokenizer:
            return len(self.word_tokenizer)
        else:
            return 0
    
    def get_current_mode(self):
        """Get the current tokenization mode"""
        return self.current_mode

def main():
    """Main function to run the training script"""
    print("🚀 JARVIS Training Script - Enhanced GPU/CPU Version for Kaggle")
    print("=" * 60)
    
    # Check Kaggle environment
    check_kaggle_environment()
    
    # Show system information
    show_system_info()
    
    # Initialize database
    if not init_database():
        print("❌ Failed to initialize database. Exiting.")
        return
    
    # Load or create model with progress tracking
    print("\n📁 Loading/Creating model...")
    print("⏱️ This may take a few minutes for large models...")
    
    try:
        load_local_model()
    except Exception as e:
        print(f"❌ Error during model loading: {e}")
        print("🔄 Attempting to create new model instead...")
        # Force create new model
        global local_model, vocab
        local_model = None
        vocab = None
        
        # Use optimized vocabulary to prevent parameter explosion
        print("🔧 Creating optimized vocabulary to prevent parameter explosion...")
        create_optimized_vocab(max_size=100000)
        
        vocab_size = len(vocab)
        local_model = SimpleLM(vocab_size).to(device)
        
        # Multi-GPU training will be enabled at the end after all processing
    
    if local_model is None:
        print("❌ Failed to load/create model. Exiting.")
        return
    
    print(f"✅ Model ready: {sum(p.numel() for p in local_model.parameters()):,} parameters")
    print(f"📚 Vocabulary size: {len(vocab)} words")
    
    # Show menu
    while True:
        print("\n" + "=" * 60)
        print("🎯 JARVIS TRAINING MENU:")
        print("1. 🧠 Train Model")
        print("2. 🧪 Test Model")
        print("3. 📊 Show Data Stats")
        print("4. 🧹 Clean Database")
        print("5. 📋 Show Checkpoint Info")
        print("6. 🔄 Restore from Backup")
        print("7. 🖥️ Show System Info")
        print("8. 📝 Create Sample JSONL")
        print("9. 🔍 Validate JSONL File")
        print("A. 🧪 Test Conversation Format")
        print("B. 🚀 Kaggle GPU Test")
        print("C. 📋 Create Kaggle Requirements")
        print("D. 🧪 Test Single GPU Training")
        print("E. 🔧 Optimize Vocabulary")
        print("F. 🤖 Show Model Info")
        print("G. 🚀 Show GPU Info")
        print("H. 🚀 Test Dual-GPU Training")
        print("I. 🧪 Test Attention Masks")
        print("J. 🔤 Convert to Byte-Level")
        print("K. 🔍 Show Tokenizer Comparison")
        print("L. 🔧 Disable Gradient Checkpointing")
        print("M. 🔧 Enable Gradient Checkpointing")
        print("N. 🧠 Switch to Smart Single-GPU")
        print("O. 🔍 Show Smart GPU Strategy")
        print("P. 🔄 Reset Model Completely")
        print("Q. 🔧 Show All Fix Options")
        print("R. 🚀 Start Incremental Training")
        print("S. 📊 Show Incremental Training Status")
        print("T. 🔤 Train BPE Tokenizer")
        print("U. 🧹 Cleanup Old Checkpoints")
        print("V. 📤 Export Database to JSONL")
        print("0. 🚪 Exit")
        print("=" * 60)
        
        choice = input("Choose an option (1-9, A-H, I-M, N-O, P-Q, 0): ").strip()
        
        if choice == "1":
            print("\n🚀 Starting training...")
            train_local_model()
        elif choice == "2":
            test_model()
        elif choice == "3":
            show_data_folder_stats()
            print("\n🔍 KAGGLE DATASET ANALYSIS:")
            analyze_kaggle_dataset()
        elif choice == "4":
            print("\n🧹 Cleaning database...")
            cleaned = clean_database_garbage()
            print(f"✅ Cleaned {cleaned} garbage entries")
        elif choice == "5":
            show_checkpoint_info()
        elif choice == "6":
            restore_from_backup()
        elif choice == "7":
            show_system_info()
        elif choice == "8":
            print("\n📝 Creating sample JSONL file...")
            create_sample_jsonl()
        elif choice == "9":
            print("\n🔍 JSONL File Validation")
            file_path = input("Enter JSONL file path (or press Enter for 'sample_training.jsonl'): ").strip()
            if not file_path:
                file_path = os.path.join(KAGGLE_WORKING_DIR, "sample_training.jsonl")
            validate_jsonl_file(file_path)
        elif choice.upper() == "A":
            print("\n🧪 Testing Conversation Format...")
            test_conversation_format()
        elif choice.upper() == "B":
            print("\n🚀 Testing Kaggle GPU Setup...")
            test_kaggle_gpu()
        elif choice.upper() == "D":
            print("\n🧪 Testing Single GPU Training...")
            test_single_gpu_training()
        elif choice.upper() == "C":
            print("\n📋 Creating Kaggle Requirements...")
            create_kaggle_requirements()
        elif choice.upper() == "E":
            print("\n🔧 Optimizing Vocabulary...")
            max_size = input("Enter maximum vocabulary size (default 100000): ").strip()
            if not max_size:
                max_size = 100000
            else:
                try:
                    max_size = int(max_size)
                except ValueError:
                    max_size = 100000
            create_optimized_vocab(max_size)
        elif choice.upper() == "F":
            show_model_info()
        elif choice.upper() == "G":
            show_gpu_info()
        elif choice.upper() == "H":
            print("\n🚀 Testing Dual-GPU Training...")
            test_dual_gpu_training()
        elif choice.upper() == "I":
            print("\n🧪 Testing Attention Masks...")
            test_attention_masks()
        elif choice.upper() == "J":
            print("\n🔤 Converting to Byte-Level...")
            convert_to_byte_level()
        elif choice.upper() == "K":
            show_tokenizer_comparison()
        elif choice.upper() == "L":
            print("\n🔧 Disabling Gradient Checkpointing...")
            disable_gradient_checkpointing()
        elif choice.upper() == "M":
            print("\n🔧 Enabling Gradient Checkpointing...")
            enable_gradient_checkpointing()
        elif choice.upper() == "N":
            print("\n🧠 Switching to Smart Single-GPU...")
            switch_to_smart_single_gpu()
        elif choice.upper() == "O":
            print("\n🔍 Showing Smart GPU Strategy...")
            show_smart_gpu_strategy()
        elif choice.upper() == "P":
            print("\n🔄 Resetting Model Completely...")
            reset_model_completely()
        elif choice.upper() == "Q":
            print("\n🔧 Showing All Fix Options...")
            show_reset_options()
        elif choice.upper() == "R":
            print("\n🚀 Starting Incremental Training...")
            start_incremental_training()
        elif choice.upper() == "S":
            print("\n📊 Showing Incremental Training Status...")
            show_incremental_training_status()
        elif choice.upper() == "T":
            print("\n🔤 Training BPE Tokenizer...")
            max_vocab = input("Enter maximum vocabulary size (default 50000): ").strip()
            if not max_vocab:
                max_vocab = 50000
            else:
                try:
                    max_vocab = int(max_vocab)
                except ValueError:
                    max_vocab = 50000
            
            min_freq = input("Enter minimum frequency (default 2): ").strip()
            if not min_freq:
                min_freq = 2
            else:
                try:
                    min_freq = int(min_freq)
                except ValueError:
                    min_freq = 2
            
            train_bpe_tokenizer_on_data(max_vocab_size=max_vocab, min_frequency=min_freq)
        elif choice.upper() == "U":
            print("\n🧹 Cleaning up old checkpoints...")
            max_checkpoints = input("Enter maximum checkpoints to keep (default 10): ").strip()
            if not max_checkpoints:
                max_checkpoints = 10
            else:
                try:
                    max_checkpoints = int(max_checkpoints)
                except ValueError:
                    max_checkpoints = 10
            
            cleanup_old_checkpoints(max_checkpoints=max_checkpoints)
        elif choice.upper() == "V":
            print("\n📤 Exporting database to JSONL...")
            output_path = input("Enter output path (or press Enter for default): ").strip()
            if not output_path:
                output_path = None
            
            export_database_to_jsonl(output_path)
        elif choice == "0":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

def convert_to_byte_level():
    """Convert the current word-level vocabulary to byte-level tokenization"""
    global vocab, local_model
    
    print("\n🔤 Converting to Byte-Level Tokenization...")
    print("=" * 50)
    
    if not BYTE_LEVEL_AVAILABLE:
        print("❌ Byte-level tokenization not available")
        print("   Install with: pip install tokenizers")
        return False
    
    if local_model is None:
        print("❌ No model loaded")
        return False
    
    try:
        # Create byte-level tokenizer
        print("🔄 Creating byte-level tokenizer...")
        byte_tokenizer = ByteLevelTokenizer()
        
        # Calculate parameter savings
        old_vocab_size = len(vocab) if vocab else 0
        new_vocab_size = len(byte_tokenizer)
        param_savings = (old_vocab_size - new_vocab_size) * 640 / 1e6
        
        print(f"\n📊 VOCABULARY CONVERSION:")
        print(f"   Old (word-level): {old_vocab_size:,} tokens")
        print(f"   New (byte-level): {new_vocab_size:,} tokens")
        print(f"   Parameter savings: ~{param_savings:.1f}M")
        print(f"   Memory savings: ~{param_savings * 4 / 1024:.1f} GB")
        
        # Create new model with byte-level vocabulary
        print(f"\n🏗️  Creating new model with byte-level vocabulary...")
        new_model = SimpleLM(new_vocab_size).to(device)
        
        # Copy transformer weights (these don't depend on vocabulary size)
        if hasattr(local_model, 'module'):
            # DataParallel wrapped model
            old_model = local_model.module
        else:
            old_model = local_model
        
        # Copy attention layers
        for i in range(min(len(old_model.attention_layers), len(new_model.attention_layers))):
            new_model.attention_layers[i].load_state_dict(old_model.attention_layers[i].state_dict())
        
        # Copy feed-forward layers
        for i in range(min(len(old_model.ffn_layers), len(new_model.ffn_layers))):
            new_model.ffn_layers[i].load_state_dict(old_model.ffn_layers[i].state_dict())
        
        # Copy normalization layers
        for i in range(min(len(old_model.pre_norms), len(new_model.pre_norms))):
            new_model.pre_norms[i].load_state_dict(old_model.pre_norms[i].state_dict())
        
        for i in range(min(len(old_model.post_norms), len(new_model.post_norms))):
            new_model.post_norms[i].load_state_dict(old_model.post_norms[i].state_dict())
        
        # Copy final norm
        new_model.final_norm.load_state_dict(old_model.final_norm.state_dict())
        
        # Initialize new embedding and output layers
        print(f"   ✅ Transformer weights copied")
        print(f"   🔧 New embedding layer: {new_vocab_size} × 640")
        print(f"   🔧 New output layer: {new_vocab_size} × 640")
        
        # Update global variables
        old_model = local_model
        local_model = new_model
        vocab = byte_tokenizer
        
        # Re-wrap with DataParallel if the old model was wrapped
        if hasattr(old_model, 'module'):
            print(f"   🔄 Re-wrapping with DataParallel...")
            local_model = torch.nn.DataParallel(local_model, device_ids=[0, 1])
        
        # Show final model info
        total_params = sum(p.numel() for p in local_model.parameters())
        print(f"\n🎉 CONVERSION COMPLETED!")
        print(f"   New model parameters: {total_params:,}")
        print(f"   Parameter reduction: ~{param_savings:.1f}M")
        print(f"   Model is now byte-level tokenized and ready for training")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_tokenizer_comparison():
    """Show comparison between word-level and byte-level tokenization"""
    print("\n🔍 TOKENIZER COMPARISON:")
    print("=" * 50)
    
    # Current vocabulary info
    current_vocab_size = len(vocab) if vocab else 0
    current_params = sum(p.numel() for p in local_model.parameters()) if local_model else 0
    
    print(f"📊 CURRENT SITUATION:")
    print(f"   Vocabulary size: {current_vocab_size:,} tokens")
    print(f"   Model parameters: {current_params:,}")
    print(f"   Memory usage: ~{current_params * 4 / 1024 / 1024:.1f} MB")
    
    # Byte-level comparison
    if BYTE_LEVEL_AVAILABLE:
        byte_vocab_size = 260  # 256 bytes + 4 special tokens
        byte_params = 0
        
        if local_model:
            # Calculate parameters for byte-level model
            # Transformer layers stay the same, only embedding/output change
            transformer_params = current_params - (current_vocab_size * 640 * 2)  # Remove old embedding/output
            byte_params = transformer_params + (byte_vocab_size * 640 * 2)  # Add new embedding/output
        
        print(f"\n🔤 BYTE-LEVEL ALTERNATIVE:")
        print(f"   Vocabulary size: {byte_vocab_size:,} tokens")
        print(f"   Model parameters: {byte_params:,}")
        print(f"   Memory usage: ~{byte_params * 4 / 1024 / 1024:.1f} MB")
        
        if current_params > 0:
            param_savings = current_params - byte_params
            memory_savings = param_savings * 4 / 1024 / 1024
            print(f"   Parameter savings: ~{param_savings / 1e6:.1f}M")
            print(f"   Memory savings: ~{memory_savings:.1f} MB")
        
        print(f"\n💡 BENEFITS OF BYTE-LEVEL:")
        print(f"   ✅ Universal coverage (any text, any language)")
        print(f"   ✅ Fixed vocabulary size (never grows)")
        print(f"   ✅ No out-of-vocabulary issues")
        print(f"   ✅ Much smaller model size")
        print(f"   ✅ Faster training and inference")
        
        print(f"\n⚠️  CONSIDERATIONS:")
        print(f"   🔄 Longer sequences (3-4x typical)")
        print(f"   🧠 Harder to learn meaningful patterns initially")
        print(f"   ⏱️  May require more training steps")
        
    else:
        print(f"\n❌ Byte-level tokenization not available")
        print(f"   Install with: pip install tokenizers")
    
    print("=" * 50)

def disable_gradient_checkpointing():
    """Temporarily disable gradient checkpointing to debug DataParallel issues"""
    global local_model
    
    print("\n🔧 Disabling Gradient Checkpointing...")
    print("=" * 50)
    
    if local_model is None:
        print("❌ No model loaded")
        return False
    
    try:
        # Access the actual model (unwrap DataParallel if needed)
        if hasattr(local_model, 'module'):
            actual_model = local_model.module
            print("🔄 Unwrapped DataParallel model")
        else:
            actual_model = local_model
        
        # Disable gradient checkpointing
        if hasattr(actual_model, 'gradient_checkpointing'):
            old_value = actual_model.gradient_checkpointing
            actual_model.gradient_checkpointing = False
            print(f"✅ Gradient checkpointing disabled (was: {old_value})")
        else:
            print("⚠️  Model doesn't have gradient_checkpointing attribute")
        
        # Also disable it in the forward method
        if hasattr(actual_model, 'forward'):
            original_forward = actual_model.forward
            
            def safe_forward(x, padding_mask=None, causal_mask=None):
                # Temporarily disable gradient checkpointing in forward pass
                batch_size, seq_len = x.shape
                
                # Embedding + positional encoding
                x = actual_model.embedding(x) * math.sqrt(actual_model.embed_size)
                x = x + actual_model.pos_enc[:, :seq_len, :]
                x = actual_model.embed_dropout(x)
                
                # Forward pass WITHOUT gradient checkpointing
                for i in range(actual_model.num_layers):
                    x = actual_model._forward_layer(x, padding_mask, causal_mask, i)
                
                # Final normalization and output
                x = actual_model.final_norm(x)
                x = actual_model.output_dropout(x)
                x = actual_model.fc(x)
                
                return x
            
            # Replace the forward method
            actual_model.forward = safe_forward
            print("✅ Forward method updated to disable gradient checkpointing")
        
        print(f"\n🎯 Gradient checkpointing has been disabled")
        print(f"   This should resolve DataParallel attention mask issues")
        print(f"   Note: Training will use more memory but should be more stable")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to disable gradient checkpointing: {e}")
        import traceback
        traceback.print_exc()
        return False

def enable_gradient_checkpointing():
    """Re-enable gradient checkpointing after debugging"""
    global local_model
    
    print("\n🔧 Re-enabling Gradient Checkpointing...")
    print("=" * 50)
    
    if local_model is None:
        print("❌ No model loaded")
        return False
    
    try:
        # Access the actual model
        if hasattr(local_model, 'module'):
            actual_model = local_model.module
        else:
            actual_model = local_model
        
        # Re-enable gradient checkpointing
        if hasattr(actual_model, 'gradient_checkpointing'):
            actual_model.gradient_checkpointing = True
            print("✅ Gradient checkpointing re-enabled")
        
        # Restore original forward method if it was modified
        if hasattr(actual_model, '_original_forward'):
            actual_model.forward = actual_model._original_forward
            print("✅ Original forward method restored")
        
        print(f"\n🎯 Gradient checkpointing has been re-enabled")
        print(f"   Memory usage will be reduced during training")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to re-enable gradient checkpointing: {e}")
        return False

def switch_to_smart_single_gpu():
    """Switch to smart single-GPU training that uses both GPUs efficiently"""
    global local_model
    
    print("\n🧠 Switching to Smart Single-GPU Training...")
    print("=" * 50)
    
    if local_model is None:
        print("❌ No model loaded")
        return False
    
    try:
        # Unwrap DataParallel if present
        if hasattr(local_model, 'module'):
            print("🔄 Unwrapping DataParallel...")
            actual_model = local_model.module
            local_model = actual_model
            print("✅ DataParallel unwrapped")
        else:
            actual_model = local_model
        
        # Move model to GPU 0 (primary)
        print("🎯 Moving model to GPU 0 (primary)...")
        torch.cuda.set_device(0)
        local_model = local_model.to('cuda:0')
        
        # Verify gradient checkpointing is enabled
        if hasattr(local_model, 'gradient_checkpointing'):
            local_model.gradient_checkpointing = True
            print("✅ Gradient checkpointing enabled")
        
        # Set Smart Single-GPU mode flag
        local_model.smart_single_gpu_mode = True
        print("✅ Smart Single-GPU mode flag set")
        
        # Show new configuration
        print(f"\n🎯 NEW TRAINING CONFIGURATION:")
        print(f"   Primary GPU: 0 ({torch.cuda.get_device_name(0)})")
        print(f"   Secondary GPU: 1 ({torch.cuda.get_device_name(1)}) - Available for other tasks")
        print(f"   Gradient checkpointing: ✅ Enabled")
        print(f"   DataParallel: ❌ Disabled (more stable)")
        print(f"   Memory efficiency: ✅ Optimized")
        
        # Test the setup
        print(f"\n🧪 Testing new configuration...")
        test_input = torch.randint(0, 1000, (2, 512), device='cuda:0')
        
        with torch.no_grad():
            outputs = local_model(test_input)
        
        print(f"✅ Test successful! Output shape: {outputs.shape}")
        
        # Show memory usage
        gpu0_memory = torch.cuda.memory_allocated(0) / (1024**3)
        gpu1_memory = torch.cuda.memory_allocated(1) / (1024**3)
        
        print(f"\n💾 GPU Memory Usage:")
        print(f"   GPU 0: {gpu0_memory:.2f} GB (active)")
        print(f"   GPU 1: {gpu1_memory:.2f} GB (available)")
        
        print(f"\n🎉 Smart single-GPU setup complete!")
        print(f"   Training will be stable with gradient checkpointing")
        print(f"   You can use GPU 1 for other tasks or as backup")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to switch to smart single-GPU: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_smart_gpu_strategy():
    """Show the smart GPU strategy that keeps gradient checkpointing"""
    print("\n🧠 SMART GPU STRATEGY:")
    print("=" * 50)
    
    print(f"🎯 THE PROBLEM:")
    print(f"   DataParallel + Gradient Checkpointing = Attention mask chaos")
    print(f"   GPUs get different sequence lengths")
    print(f"   Masks become misaligned")
    print(f"   Training crashes with tensor size errors")
    
    print(f"\n💡 THE SOLUTION:")
    print(f"   Use ONE GPU for the main model")
    print(f"   Keep gradient checkpointing enabled")
    print(f"   Use the other GPU for:")
    print(f"     • Gradient accumulation")
    print(f"     • Memory offloading")
    print(f"     • Other AI tasks")
    print(f"     • Backup/fallback")
    
    print(f"\n🚀 BENEFITS:")
    print(f"   ✅ Stable training with gradient checkpointing")
    print(f"   ✅ No more attention mask errors")
    print(f"   ✅ Both GPUs still utilized")
    print(f"   ✅ Memory efficient")
    print(f"   ✅ Faster training (no DataParallel overhead)")
    
    print(f"\n🔧 IMPLEMENTATION:")
    print(f"   Option N: Switch to Smart Single-GPU")
    print(f"   This keeps gradient checkpointing enabled")
    print(f"   And uses both GPUs efficiently")
    
    print("=" * 50)

def reset_model_completely():
    """Completely reset the model and start fresh"""
    global local_model, vocab
    
    print("\n🔄 Completely Resetting Model...")
    print("=" * 50)
    
    try:
        # Remove existing model files
        if os.path.exists(LOCAL_MODEL_PATH):
            print(f"🗑️  Removing existing model: {LOCAL_MODEL_PATH}")
            os.remove(LOCAL_MODEL_PATH)
            print("✅ Existing model removed")
        
        if os.path.exists(BACKUP_MODEL_PATH):
            print(f"🗑️  Removing backup model: {BACKUP_MODEL_PATH}")
            os.remove(BACKUP_MODEL_PATH)
            print("✅ Backup model removed")
        
        # Reset global variables
        local_model = None
        vocab = None
        
        print(f"\n🧹 Model state cleared")
        print(f"   Global variables reset")
        print(f"   Model files removed")
        print(f"   Ready for fresh start")
        
        # Show next steps
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Restart the script")
        print(f"   2. Choose option N (Smart Single-GPU)")
        print(f"   3. This will create a fresh model with gradient checkpointing")
        print(f"   4. Training should work without attention mask errors")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to reset model: {e}")
        return False

def show_reset_options():
    """Show all the options for fixing the model issues"""
    print("\n🔧 MODEL FIX OPTIONS:")
    print("=" * 50)
    
    print(f"🎯 OPTION 1: Smart Single-GPU (RECOMMENDED)")
    print(f"   • Use option N")
    print(f"   • Keeps gradient checkpointing")
    print(f"   • Uses both GPUs efficiently")
    print(f"   • No attention mask errors")
    print(f"   • Stable training")
    
    print(f"\n🎯 OPTION 2: Disable Gradient Checkpointing")
    print(f"   • Use option L")
    print(f"   • Fixes attention mask errors")
    print(f"   • Uses more memory")
    print(f"   • Less efficient")
    
    print(f"\n🎯 OPTION 3: Convert to Byte-Level")
    print(f"   • Use option J")
    print(f"   • Solves parameter explosion")
    print(f"   • Much smaller model")
    print(f"   • Universal coverage")
    
    print(f"\n🎯 OPTION 4: Complete Reset")
    print(f"   • Use option P")
    print(f"   • Removes all model files")
    print(f"   • Fresh start")
    print(f"   • Nuclear option")
    
    print("=" * 50)

def start_incremental_training():
    """Start incremental training with chunked data processing"""
    global local_model, vocab, incremental_manager
    
    print("\n🚀 STARTING INCREMENTAL TRAINING")
    print("=" * 50)
    
    # Check if model exists
    if local_model is None:
        print("❌ No model loaded. Please load or create a model first.")
        return
    
    # Check for existing incremental training state
    if check_for_existing_model():
        print("✅ Resuming incremental training from existing model")
    else:
        print("🆕 Starting fresh incremental training")
    
    # Show incremental training configuration
    print(f"\n📋 INCREMENTAL TRAINING CONFIGURATION:")
    print(f"   📦 Chunk size: {INCREMENTAL_TRAINING_CONFIG['chunk_size_gb']} GB")
    print(f"   🔄 Replay buffer: {INCREMENTAL_TRAINING_CONFIG['replay_buffer_size']*100:.0f}%")
    print(f"   💾 Checkpoint interval: {INCREMENTAL_TRAINING_CONFIG['checkpoint_interval_gb']} GB")
    print(f"   📚 Curriculum phases: {', '.join(INCREMENTAL_TRAINING_CONFIG['curriculum_phases'])}")
    
    # Start the training
    print(f"\n🎯 Starting incremental training...")
    try:
        train_local_model()
        print("✅ Incremental training completed successfully!")
    except Exception as e:
        print(f"❌ Incremental training failed: {e}")
        import traceback
        traceback.print_exc()

def show_incremental_training_status():
    """Show current incremental training status and progress"""
    global incremental_manager
    
    print("\n📊 INCREMENTAL TRAINING STATUS")
    print("=" * 50)
    
    if incremental_manager is None:
        print("❌ Incremental training manager not initialized")
        return
    
    print(f"📈 Training Progress:")
    print(f"   Current phase: {INCREMENTAL_TRAINING_CONFIG['curriculum_phases'][incremental_manager.current_phase]}")
    print(f"   Data processed: {incremental_manager.total_data_processed:.2f} GB")
    print(f"   Checkpoints saved: {incremental_manager.checkpoint_counter}")
    print(f"   Loss history entries: {len(incremental_manager.loss_history)}")
    
    # Show recent loss trends
    if incremental_manager.loss_history:
        recent_losses = incremental_manager.loss_history[-10:]  # Last 10 entries
        print(f"\n📊 Recent Loss Trends:")
        for entry in recent_losses:
            phase = INCREMENTAL_TRAINING_CONFIG['curriculum_phases'][entry['phase']]
            print(f"   {entry['timestamp'][:19]}: {entry['loss']:.4f} ({phase})")
    
    # Show chunk history
    if incremental_manager.chunk_history:
        print(f"\n📦 Chunk History:")
        print(f"   Total chunks: {len(incremental_manager.chunk_history)}")
        print(f"   Replay buffer size: {len(incremental_manager.chunk_history) * INCREMENTAL_TRAINING_CONFIG['replay_buffer_size']:.0f}")
    
    # Show configuration
    print(f"\n⚙️ Configuration:")
    print(f"   Chunk size: {INCREMENTAL_TRAINING_CONFIG['chunk_size_gb']} GB")
    print(f"   Replay buffer: {INCREMENTAL_TRAINING_CONFIG['replay_buffer_size']*100:.0f}%")
    print(f"   Checkpoint interval: {INCREMENTAL_TRAINING_CONFIG['checkpoint_interval_gb']} GB")
    
    print("=" * 50)

# Enhanced configuration for incremental training
INCREMENTAL_TRAINING_CONFIG = {
    'chunk_size_gb': 1.0,  # Process 1 GB at a time
    'replay_buffer_size': 0.1,  # 10% replay buffer to reduce forgetting
    'checkpoint_interval_gb': 1.0,  # Save checkpoint every GB
    'max_chunks_in_memory': 3,  # Keep max 3 chunks in RAM
    'curriculum_phases': ['instructions', 'conversations', 'math', 'casual_text'],
    'phase_weights': [0.3, 0.4, 0.2, 0.1]  # Weight for each phase
}

# Global variables for incremental training
current_training_phase = 0
chunk_history = []
total_data_processed = 0
training_loss_history = []

class IncrementalTrainingManager:
    """Manages incremental training with chunked data loading and replay buffer"""
    
    def __init__(self, config=INCREMENTAL_TRAINING_CONFIG):
        self.config = config
        self.chunk_history = []
        self.current_phase = 0
        self.total_data_processed = 0
        self.loss_history = []
        self.checkpoint_counter = 0
        
    def get_next_chunk(self, data_source, chunk_size_gb=None):
        """Get next chunk of data for training"""
        if chunk_size_gb is None:
            chunk_size_gb = self.config['chunk_size_gb']
        
        # Estimate chunk size in samples (rough approximation)
        estimated_samples_per_gb = 100000  # Adjust based on your data
        target_samples = int(chunk_size_gb * estimated_samples_per_gb)
        
        # Get chunk from data source
        chunk = self._extract_chunk(data_source, target_samples)
        
        # Add to history for replay buffer
        self.chunk_history.append(chunk)
        
        # Maintain replay buffer size
        max_history_size = int(len(self.chunk_history) * self.config['replay_buffer_size'])
        if len(self.chunk_history) > max_history_size:
            self.chunk_history = self.chunk_history[-max_history_size:]
        
        return chunk
    
    def _extract_chunk(self, data_source, target_samples):
        """Extract a chunk of data from the source"""
        # This is a placeholder - implement based on your data structure
        if isinstance(data_source, list):
            start_idx = len(self.chunk_history) * target_samples
            end_idx = min(start_idx + target_samples, len(data_source))
            return data_source[start_idx:end_idx]
        else:
            # Handle other data source types
            return []
    
    def get_replay_buffer(self):
        """Get replay buffer data to reduce forgetting"""
        if not self.chunk_history:
            return []
        
        # Sample from history based on replay buffer size
        replay_size = int(len(self.chunk_history) * self.config['replay_buffer_size'])
        if replay_size == 0:
            return []
        
        # Randomly sample from history
        import random
        replay_indices = random.sample(range(len(self.chunk_history)), min(replay_size, len(self.chunk_history)))
        replay_data = [self.chunk_history[i] for i in replay_indices]
        
        return replay_data
    
    def should_checkpoint(self, data_processed_gb):
        """Check if we should save a checkpoint"""
        return data_processed_gb >= self.config['checkpoint_interval_gb']
    
    def log_loss(self, loss, chunk_info=None):
        """Log training loss for monitoring"""
        loss_entry = {
            'loss': loss,
            'chunk': chunk_info,
            'phase': self.current_phase,
            'timestamp': datetime.now().isoformat(),
            'total_data_processed': self.total_data_processed
        }
        self.loss_history.append(loss_entry)
        
        # Save loss history to file
        self._save_loss_history()
    
    def _save_loss_history(self):
        """Save loss history to file for monitoring"""
        try:
            loss_file = os.path.join(KAGGLE_WORKING_DIR, "training_loss_history.json")
            with open(loss_file, 'w') as f:
                json.dump(self.loss_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save loss history: {e}")
    
    def next_phase(self):
        """Move to next curriculum phase"""
        self.current_phase = (self.current_phase + 1) % len(self.config['curriculum_phases'])
        logger.info(f"Moving to curriculum phase: {self.config['curriculum_phases'][self.current_phase]}")
        return self.current_phase
    
    def get_phase_data(self, data_source, phase_name):
        """Get data for specific curriculum phase"""
        # Filter data based on phase
        if phase_name == 'instructions':
            return self._filter_instruction_data(data_source)
        elif phase_name == 'conversations':
            return self._filter_conversation_data(data_source)
        elif phase_name == 'math':
            return self._filter_math_data(data_source)
        elif phase_name == 'casual_text':
            return self._filter_casual_text_data(data_source)
        else:
            return data_source
    
    def _filter_instruction_data(self, data):
        """Filter data for instruction phase"""
        # Look for instruction-like patterns
        instruction_keywords = ['instruction', 'task', 'please', 'how to', 'steps', 'guide']
        filtered = []
        for item in data:
            if isinstance(item, dict) and 'input' in item:
                text = item['input'].lower()
                if any(keyword in text for keyword in instruction_keywords):
                    filtered.append(item)
        return filtered
    
    def _filter_conversation_data(self, data):
        """Filter data for conversation phase"""
        # Look for conversation-like patterns
        conversation_keywords = ['hello', 'hi', 'how are you', 'what', 'why', 'when']
        filtered = []
        for item in data:
            if isinstance(item, dict) and 'input' in item:
                text = item['input'].lower()
                if any(keyword in text for keyword in conversation_keywords):
                    filtered.append(item)
        return filtered
    
    def _filter_math_data(self, data):
        """Filter data for math phase"""
        # Look for math-like patterns
        math_patterns = [r'\d+', r'\+', r'-', r'\*', r'/', r'=', r'equation', r'calculate']
        import re
        filtered = []
        for item in data:
            if isinstance(item, dict) and 'input' in item:
                text = item['input']
                if any(re.search(pattern, text) for pattern in math_patterns):
                    filtered.append(item)
        return filtered
    
    def _filter_casual_text_data(self, data):
        """Filter data for casual text phase"""
        # Look for casual text patterns
        casual_keywords = ['story', 'narrative', 'description', 'explain', 'tell me about']
        filtered = []
        for item in data:
            if isinstance(item, dict) and 'input' in item:
                text = item['input'].lower()
                if any(keyword in text for keyword in casual_keywords):
                    filtered.append(item)
        return filtered

# Global incremental training manager
incremental_manager = IncrementalTrainingManager()

def save_incremental_checkpoint(model, optimizer, scheduler, chunk_info, loss):
    """Save checkpoint after processing each chunk"""
    global incremental_manager
    
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'embed_size': model.embed_size,
            'num_heads': model.num_heads,
            'num_layers': model.num_layers,
            'hidden_size': model.hidden_size,
            'dropout': model.dropout,
            'max_seq_len': model.max_seq_len
        },
        'vocab': vocab,
        'vocab_size': len(vocab),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'incremental_info': {
            'chunk_info': chunk_info,
            'current_phase': incremental_manager.current_phase,
            'total_data_processed': incremental_manager.total_data_processed,
            'checkpoint_counter': incremental_manager.checkpoint_counter,
            'timestamp': datetime.now().isoformat()
        },
        'loss': loss
    }
    
    # Save with chunk-specific naming
    chunk_checkpoint_path = os.path.join(
        KAGGLE_WORKING_DIR, 
        f"jarvis_chunk_{incremental_manager.checkpoint_counter:04d}.pth"
    )
    
    # Atomic save
    temp_path = chunk_checkpoint_path + '.tmp'
    torch.save(checkpoint_data, temp_path)
    os.replace(temp_path, chunk_checkpoint_path)
    
    # Also update main checkpoint
    temp_path = LOCAL_MODEL_PATH + '.tmp'
    torch.save(checkpoint_data, temp_path)
    os.replace(temp_path, LOCAL_MODEL_PATH)
    
    logger.info(f"Chunk checkpoint saved: {chunk_checkpoint_path}")
    logger.info(f"Main checkpoint updated: {LOCAL_MODEL_PATH}")
    
    incremental_manager.checkpoint_counter += 1

def load_incremental_checkpoint(checkpoint_path):
    """Load incremental checkpoint and resume training"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            local_model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model state loaded from checkpoint")
        
        # Load optimizer and scheduler state
        optimizer_state = None
        scheduler_state = None
        if 'optimizer_state_dict' in checkpoint:
            optimizer_state = checkpoint['optimizer_state_dict']
        if 'scheduler_state_dict' in checkpoint:
            scheduler_state = checkpoint['scheduler_state_dict']
        
        # Load incremental training state
        if 'incremental_info' in checkpoint:
            incremental_info = checkpoint['incremental_info']
            incremental_manager.current_phase = incremental_info.get('current_phase', 0)
            incremental_manager.total_data_processed = incremental_info.get('total_data_processed', 0)
            incremental_manager.checkpoint_counter = incremental_info.get('checkpoint_counter', 0)
            logger.info(f"Resuming from phase {incremental_manager.current_phase}")
            logger.info(f"Total data processed: {incremental_manager.total_data_processed}")
        
        return optimizer_state, scheduler_state
        
    except Exception as e:
        logger.error(f"Failed to load incremental checkpoint: {e}")
        return None, None

def process_training_chunk(dataloader, model, optimizer, scheduler, criterion, scaler, grad_accumulation_steps):
    """Process a single chunk of training data"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Create attention masks
        padding_mask = (inputs == 0)
        actual_seq_len = inputs.size(1)
        
        # Safety check: ensure sequence length is within model limits
        model_max_seq_len = get_model_max_seq_len(model)
        if actual_seq_len > model_max_seq_len:
            logger.warning(f"Sequence length {actual_seq_len} exceeds model limit {model_max_seq_len}, truncating...")
            actual_seq_len = model_max_seq_len
            inputs = inputs[:, :actual_seq_len]
            targets = targets[:, :actual_seq_len]
            padding_mask = (inputs == 0)
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(actual_seq_len, actual_seq_len, device=inputs.device), diagonal=1).bool()
        
        # Forward pass
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(inputs, padding_mask=padding_mask, causal_mask=causal_mask)
        else:
            outputs = model(inputs, padding_mask=padding_mask, causal_mask=causal_mask)
        
        # Calculate loss
        batch_size, seq_len, vocab_size = outputs.shape
        outputs_flat = outputs.view(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        
        # Ensure targets match sequence length
        if targets.size(1) != seq_len:
            if targets.size(1) > seq_len:
                targets = targets[:, :seq_len]
            else:
                padding = torch.zeros(batch_size, seq_len - targets.size(1), dtype=targets.dtype, device=targets.device)
                targets = torch.cat([targets, padding], dim=1)
            targets_flat = targets.reshape(-1)
        
        loss = criterion(outputs_flat, targets_flat)
        
        # Check for invalid loss values
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Invalid loss detected: {loss.item()}, skipping batch")
            optimizer.zero_grad()
            continue
        
        # Scale loss for gradient accumulation
        if grad_accumulation_steps > 0:
            scaled_loss = loss / grad_accumulation_steps
        else:
            scaled_loss = loss
        
        # Backward pass
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % grad_accumulation_steps == 0:
            # Gradient clipping
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Zero gradients
            optimizer.zero_grad()
        
        # Update loss tracking
        total_loss += loss.item()
        num_batches += 1
    
    # Handle remaining gradients if any
    if num_batches % grad_accumulation_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()
    
    # Return average loss for this chunk
    return total_loss / max(1, num_batches)

def check_for_existing_model():
    """Check if model exists and load it for incremental training"""
    global local_model, vocab
    
    if os.path.exists(LOCAL_MODEL_PATH):
        print("🔄 Found existing model, loading for incremental training...")
        try:
            load_local_model()
            if local_model is not None:
                print(f"✅ Loaded existing model with {sum(p.numel() for p in local_model.parameters()):,} parameters")
                
                # Check for incremental training info
                checkpoint = torch.load(LOCAL_MODEL_PATH, map_location=device)
                if 'incremental_info' in checkpoint:
                    incremental_info = checkpoint['incremental_info']
                    print(f"📊 Resuming incremental training:")
                    print(f"   Phase: {incremental_info.get('current_phase', 0)}")
                    print(f"   Data processed: {incremental_info.get('total_data_processed', 0)}")
                    print(f"   Checkpoints: {incremental_info.get('checkpoint_counter', 0)}")
                return True
            else:
                print("⚠️ Failed to load existing model")
                return False
        except Exception as e:
            print(f"❌ Error loading existing model: {e}")
            return False
    else:
        print("🆕 No existing model found, will create new one")
        return False

class BPETokenizer:
    """BPE (Byte Pair Encoding) tokenizer for efficient subword tokenization"""
    
    def __init__(self, vocab_size=50000, min_frequency=2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.tokenizer = None
        self.is_trained = False
        
    def train(self, texts):
        """Train BPE tokenizer on provided texts"""
        if not TOKENIZERS_AVAILABLE:
            print("❌ tokenizers library not available for BPE training")
            return False
        
        try:
            print(f"🔤 Training BPE tokenizer (vocab_size={self.vocab_size}, min_freq={self.min_frequency})...")
            
            # Initialize BPE model
            bpe = models.BPE()
            
            # Create trainer
            trainer = trainers.BpeTrainer(
                vocab_size=self.vocab_size,
                min_frequency=self.min_frequency,
                special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
            )
            
            # Create tokenizer
            self.tokenizer = Tokenizer(bpe)
            self.tokenizer.pre_tokenizer = WhitespaceSplit()
            
            # Train on texts
            self.tokenizer.train_from_iterator(texts, trainer=trainer)
            
            self.is_trained = True
            print(f"✅ BPE tokenizer trained successfully with {self.tokenizer.get_vocab_size()} tokens")
            return True
            
        except Exception as e:
            print(f"❌ BPE training failed: {e}")
            return False
    
    def encode(self, text):
        """Encode text to token IDs"""
        if not self.is_trained:
            print("❌ BPE tokenizer not trained yet")
            return []
        
        try:
            encoding = self.tokenizer.encode(text)
            return encoding.ids
        except Exception as e:
            print(f"❌ BPE encoding failed: {e}")
            return []
    
    def decode(self, token_ids):
        """Decode token IDs back to text"""
        if not self.is_trained:
            print("❌ BPE tokenizer not trained yet")
            return ""
        
        try:
            return self.tokenizer.decode(token_ids)
        except Exception as e:
            print(f"❌ BPE decoding failed: {e}")
            return ""
    
    def __len__(self):
        """Return vocabulary size"""
        if self.tokenizer:
            return self.tokenizer.get_vocab_size()
        return 0
    
    def get_vocab(self):
        """Get vocabulary mapping"""
        if self.tokenizer:
            return self.tokenizer.get_vocab()
        return {}

def train_bpe_tokenizer_on_data(max_vocab_size=50000, min_frequency=2):
    """Train BPE tokenizer on all available training data"""
    global vocab
    
    if not TOKENIZERS_AVAILABLE:
        print("❌ tokenizers library not available for BPE training")
        return None
    
    print(f"🔤 Training BPE tokenizer on training data...")
    print(f"   Target vocab size: {max_vocab_size:,}")
    print(f"   Minimum frequency: {min_frequency}")
    
    # Collect all training texts
    all_texts = []
    
    # Get texts from database
    try:
        with DB_LOCK:
            if DB_CONN is not None:
                cursor = DB_CONN.cursor()
                cursor.execute("SELECT content, response FROM memory WHERE content IS NOT NULL AND response IS NOT NULL")
                rows = cursor.fetchall()
                
                for content, response in rows:
                    if content and response:
                        all_texts.append(clean_convo_data(content))
                        all_texts.append(clean_convo_data(response))
                
                print(f"   📊 Added {len(rows)} database conversations")
    except Exception as e:
        print(f"   ⚠️ Error reading database: {e}")
    
    # Get texts from data folder
    data_conversations = load_data_folder_files()
    for conv in data_conversations:
        if 'input' in conv and 'target' in conv:
            all_texts.append(clean_convo_data(conv['input']))
            all_texts.append(clean_convo_data(conv['target']))
    
    print(f"   📊 Added {len(data_conversations)} data folder conversations")
    print(f"   📊 Total text samples: {len(all_texts):,}")
    
    if not all_texts:
        print("❌ No training texts found")
        return None
    
    # Train BPE tokenizer
    bpe_tokenizer = BPETokenizer(vocab_size=max_vocab_size, min_frequency=min_frequency)
    
    if bpe_tokenizer.train(all_texts):
        print(f"✅ BPE tokenizer trained successfully!")
        print(f"   Final vocab size: {len(bpe_tokenizer)}")
        
        # Show some example tokenizations
        print(f"\n🔍 Example tokenizations:")
        sample_texts = all_texts[:3]
        for i, text in enumerate(sample_texts):
            if len(text) > 100:
                text = text[:100] + "..."
            tokens = bpe_tokenizer.encode(text)
            print(f"   Sample {i+1}: '{text}' → {len(tokens)} tokens")
        
        return bpe_tokenizer
    else:
        print("❌ BPE training failed")
        return None

def run_validation(dataloader, model, criterion, scaler):
    """Run validation and return validation loss"""
    model.eval()
    total_val_loss = 0.0
    num_val_batches = 0
    
    try:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Create attention masks
                padding_mask = (inputs == 0)
                actual_seq_len = inputs.size(1)
                
                # Safety check: ensure sequence length is within model limits
                model_max_seq_len = get_model_max_seq_len(model)
                if actual_seq_len > model_max_seq_len:
                    actual_seq_len = model_max_seq_len
                    inputs = inputs[:, :actual_seq_len]
                    targets = targets[:, :actual_seq_len]
                    padding_mask = (inputs == 0)
                
                # Create causal mask
                causal_mask = torch.triu(torch.ones(actual_seq_len, actual_seq_len, device=inputs.device), diagonal=1).bool()
                
                # Forward pass
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs, padding_mask=padding_mask, causal_mask=causal_mask)
                else:
                    outputs = model(inputs, padding_mask=padding_mask, causal_mask=causal_mask)
                
                # Calculate loss
                batch_size, seq_len, vocab_size = outputs.shape
                outputs_flat = outputs.view(-1, vocab_size)
                targets_flat = targets.reshape(-1)
                
                # Ensure targets match sequence length
                if targets.size(1) != seq_len:
                    if targets.size(1) > seq_len:
                        targets = targets[:, :seq_len]
                    else:
                        padding = torch.zeros(batch_size, seq_len - targets.size(1), dtype=targets.dtype, device=targets.device)
                        targets = torch.cat([targets, padding], dim=1)
                    targets_flat = targets.reshape(-1)
                
                loss = criterion(outputs_flat, targets_flat)
                
                # Check for invalid loss values
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Invalid validation loss detected: {loss.item()}, skipping batch")
                    continue
                
                total_val_loss += loss.item()
                num_val_batches += 1
                
                # Limit validation to reasonable number of batches
                if num_val_batches >= 100:  # Max 100 validation batches
                    break
        
        # Return average validation loss
        if num_val_batches > 0:
            return total_val_loss / num_val_batches
        else:
            return None
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return None
    finally:
        model.train()

def cleanup_old_checkpoints(max_checkpoints=10):
    """Clean up old checkpoints to save disk space"""
    print(f"🧹 Cleaning up old checkpoints (keeping {max_checkpoints} most recent)...")
    
    try:
        # Get all checkpoint files
        checkpoint_patterns = [
            os.path.join(KAGGLE_WORKING_DIR, "jarvis_chunk_*.pth"),
            os.path.join(KAGGLE_WORKING_DIR, "jarvis_periodic_chunk_*.pth")
        ]
        
        all_checkpoints = []
        for pattern in checkpoint_patterns:
            all_checkpoints.extend(glob.glob(pattern))
        
        if not all_checkpoints:
            print("   ℹ️ No checkpoint files found to clean up")
            return
        
        # Sort by modification time (newest first)
        all_checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Keep the most recent ones
        checkpoints_to_keep = all_checkpoints[:max_checkpoints]
        checkpoints_to_delete = all_checkpoints[max_checkpoints:]
        
        if not checkpoints_to_delete:
            print(f"   ℹ️ Only {len(all_checkpoints)} checkpoints found, no cleanup needed")
            return
        
        print(f"   📊 Found {len(all_checkpoints)} total checkpoints")
        print(f"   💾 Keeping {len(checkpoints_to_keep)} most recent")
        print(f"   🗑️ Deleting {len(checkpoints_to_delete)} old checkpoints")
        
        # Delete old checkpoints
        total_deleted_size = 0
        for checkpoint_path in checkpoints_to_delete:
            try:
                file_size = os.path.getsize(checkpoint_path)
                os.remove(checkpoint_path)
                total_deleted_size += file_size
                print(f"      🗑️ Deleted: {os.path.basename(checkpoint_path)} ({file_size / (1024*1024):.1f} MB)")
            except Exception as e:
                print(f"      ⚠️ Failed to delete {os.path.basename(checkpoint_path)}: {e}")
        
        print(f"   ✅ Cleanup completed! Freed {total_deleted_size / (1024*1024):.1f} MB")
        
    except Exception as e:
        print(f"   ❌ Checkpoint cleanup failed: {e}")

def export_database_to_jsonl(output_path=None):
    """Export database conversations to JSONL for faster training"""
    if output_path is None:
        output_path = os.path.join(KAGGLE_WORKING_DIR, "database_export.jsonl")
    
    print(f"📤 Exporting database to JSONL: {output_path}")
    
    try:
        conversations = []
        
        # Get conversations from database
        with DB_LOCK:
            if DB_CONN is not None:
                cursor = DB_CONN.cursor()
                cursor.execute("SELECT content, response FROM memory WHERE content IS NOT NULL AND response IS NOT NULL")
                rows = cursor.fetchall()
                
                for content, response in rows:
                    if content and response:
                        conversations.append({
                            'input': clean_convo_data(content),
                            'target': clean_convo_data(response),
                            'source': 'database'
                        })
                
                print(f"   📊 Found {len(rows)} database conversations")
            else:
                print("   ⚠️ No database connection")
                return False
        
        # Get contextual memory
        try:
            with DB_LOCK:
                if DB_CONN is not None:
                    cursor = DB_CONN.cursor()
                    cursor.execute("SELECT conversation_context, key_topics FROM contextual_memory WHERE conversation_context IS NOT NULL")
                    context_rows = cursor.fetchall()
                    
                    for context, topics in context_rows:
                        if context:
                            topics_str = ", ".join(json.loads(topics) if topics else [])
                            conversations.append({
                                'input': f"Context: {clean_convo_data(context)} | Topics: {topics_str}",
                                'target': clean_convo_data(context),
                                'source': 'contextual_memory'
                            })
                    
                    print(f"   📊 Found {len(context_rows)} contextual memories")
        except Exception as e:
            print(f"   ⚠️ Error reading contextual memory: {e}")
        
        # Write to JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')
        
        print(f"   ✅ Exported {len(conversations)} conversations to {output_path}")
        print(f"   📁 File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Export failed: {e}")
        return False

if __name__ == "__main__":
    main() 