import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_bart import (
    PretrainedBartModel,  
    LayerNorm, 
    EncoderLayer, 
    DecoderLayer, 
    LearnedPositionalEmbedding,
    _prepare_bart_decoder_inputs,
    _make_linear_from_emb
)

class ParaBart(PretrainedBartModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.shared = nn.Embedding(config.vocab_size, config.d_model, config.pad_token_id)

        self.encoder = ParaBartEncoder(config, self.shared)
        self.decoder = ParaBartDecoder(config, self.shared)
                
        self.linear = nn.Linear(config.d_model, config.vocab_size)
        
        self.adversary = Discriminator(config)
        
        self.init_weights()

    def forward(
        self,
        input_ids,      
        decoder_input_ids,
        attention_mask=None,
        decoder_padding_mask=None,
        encoder_outputs=None,
        return_encoder_outputs=False,
    ):
        if attention_mask is None:
            attention_mask = input_ids == self.config.pad_token_id
        
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)
            
        if return_encoder_outputs:
            return encoder_outputs
        
        assert encoder_outputs is not None
        assert decoder_input_ids is not None

        decoder_input_ids = decoder_input_ids[:, :-1]
                
        _, decoder_padding_mask, decoder_causal_mask = _prepare_bart_decoder_inputs(
            self.config,
            input_ids=None,
            decoder_input_ids=decoder_input_ids,
            decoder_padding_mask=decoder_padding_mask,
            causal_mask_dtype=self.shared.weight.dtype,
        )    

        attention_mask2 = torch.cat((torch.zeros(input_ids.shape[0], 1).bool().cuda(), attention_mask[:, self.config.max_sent_len+2:]), dim=1)
           
        # decoder
        decoder_outputs = self.decoder(
            decoder_input_ids,
            torch.cat((encoder_outputs[1], encoder_outputs[0][:, self.config.max_sent_len+2:]), dim=1),           
            decoder_padding_mask=decoder_padding_mask,
            decoder_causal_mask=decoder_causal_mask,
            encoder_attention_mask=attention_mask2,
        )[0]
        
       
        batch_size = decoder_outputs.shape[0]
        outputs = self.linear(decoder_outputs.contiguous().view(-1, self.config.d_model))
        outputs = outputs.view(batch_size, -1, self.config.vocab_size)
        
        # discriminator
        for p in self.adversary.parameters():
            p.required_grad=False
        adv_outputs = self.adversary(encoder_outputs[1])        
        
        return outputs, adv_outputs
    
    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, use_cache, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        encoder_outputs = past[0]
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "decoder_input_ids": torch.cat((decoder_input_ids, torch.zeros((decoder_input_ids.shape[0], 1), dtype=torch.long).cuda()), 1),
            "attention_mask": attention_mask,
        }

    def get_encoder(self):
        return self.encoder

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.shared)
    
    def get_input_embeddings(self):
        return self.shared
    
    @staticmethod
    def _reorder_cache(past, beam_idx):
        enc_out = past[0][0]

        new_enc_out = enc_out.index_select(0, beam_idx)

        past = ((new_enc_out, ), )
        return past

    def forward_adv(
        self,
        input_token_ids,      
        attention_mask=None,
        decoder_padding_mask=None
    ):
        for p in self.adversary.parameters():
            p.required_grad=True
        sent_embeds = self.encoder.embed(input_token_ids, attention_mask=attention_mask).detach()
        adv_outputs = self.adversary(sent_embeds)

        return adv_outputs


class ParaBartEncoder(nn.Module):
    def __init__(self, config, embed_tokens):
        super().__init__()
        self.config = config

        self.dropout = config.dropout
        self.embed_tokens = embed_tokens
                
        self.embed_synt = nn.Embedding(77, config.d_model, config.pad_token_id)       
        self.embed_synt.weight.data.normal_(mean=0.0, std=config.init_std)
        self.embed_synt.weight.data[config.pad_token_id].zero_()

        self.embed_positions = LearnedPositionalEmbedding(
            config.max_position_embeddings, config.d_model, config.pad_token_id, config.extra_pos_embeddings
        )
        
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.synt_layers = nn.ModuleList([EncoderLayer(config) for _ in range(1)])

        self.layernorm_embedding = LayerNorm(config.d_model) 

        self.synt_layernorm_embedding = LayerNorm(config.d_model)
        
        self.pooling = MeanPooling(config)
        

    def forward(self, input_ids, attention_mask): 
        
        input_token_ids, input_synt_ids = torch.split(input_ids, [self.config.max_sent_len+2, self.config.max_synt_len+2], dim=1)
        input_token_mask, input_synt_mask = torch.split(attention_mask, [self.config.max_sent_len+2, self.config.max_synt_len+2], dim=1)
        
        x = self.forward_token(input_token_ids, input_token_mask)
        y = self.forward_synt(input_synt_ids, input_synt_mask)
                
        encoder_outputs = torch.cat((x,y), dim=1)

        sent_embeds = self.pooling(x, input_token_ids)

        return encoder_outputs, sent_embeds
    
    def forward_token(self, input_token_ids, attention_mask):
        if self.training:
            drop_mask = torch.bernoulli(self.config.word_dropout*torch.ones(input_token_ids.shape)).bool().cuda()
            input_token_ids = input_token_ids.masked_fill(drop_mask, 50264)
               
        input_token_embeds = self.embed_tokens(input_token_ids) + self.embed_positions(input_token_ids)
        x = self.layernorm_embedding(input_token_embeds)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = x.transpose(0, 1)
        
        for encoder_layer in self.layers:
            x, _ = encoder_layer(x, encoder_padding_mask=attention_mask)
            
        x = x.transpose(0, 1)
        return x
        
    def forward_synt(self, input_synt_ids, attention_mask):
        input_synt_embeds = self.embed_synt(input_synt_ids) + self.embed_positions(input_synt_ids)        
        y = self.synt_layernorm_embedding(input_synt_embeds)        
        y = F.dropout(y, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        y = y.transpose(0, 1)
            
        for encoder_synt_layer in self.synt_layers:
            y, _ = encoder_synt_layer(y, encoder_padding_mask=attention_mask)

        # T x B x C -> B x T x C
        y = y.transpose(0, 1)
        return y
        

    def embed(self, input_token_ids, attention_mask=None, pool='mean'):
        if attention_mask is None:
            attention_mask = input_token_ids == self.config.pad_token_id
            
        x = self.forward_token(input_token_ids, attention_mask)
        
        sent_embeds = self.pooling(x, input_token_ids)
        return sent_embeds
            
class MeanPooling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def forward(self, x, input_token_ids):
        mask = input_token_ids != self.config.pad_token_id
        mean_mask = mask.float()/mask.float().sum(1, keepdim=True)
        x = (x*mean_mask.unsqueeze(2)).sum(1, keepdim=True)
        return x


class ParaBartDecoder(nn.Module):
    def __init__(self, config, embed_tokens):
        super().__init__()
        
        self.dropout = config.dropout
        
        self.embed_tokens = embed_tokens
        
        self.embed_positions = LearnedPositionalEmbedding(
            config.max_position_embeddings, config.d_model, config.pad_token_id, config.extra_pos_embeddings
        )
        
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(1)]) 
        self.layernorm_embedding = LayerNorm(config.d_model)

    def forward(
        self, 
        decoder_input_ids, 
        encoder_hidden_states,  
        decoder_padding_mask, 
        decoder_causal_mask,  
        encoder_attention_mask
    ):        
		
        x = self.embed_tokens(decoder_input_ids) + self.embed_positions(decoder_input_ids)
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        for idx, decoder_layer in enumerate(self.layers):
            x, _, _ = decoder_layer(
                x, 
                encoder_hidden_states,
                encoder_attn_mask=encoder_attention_mask,
                decoder_padding_mask=decoder_padding_mask,
                causal_mask=decoder_causal_mask)

        x = x.transpose(0, 1)
       
        return x,
    
    
class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sent_layernorm_embedding = LayerNorm(config.d_model, elementwise_affine=False)
        self.adv = nn.Linear(config.d_model, 74)
        
    def forward(self, sent_embeds):
        x = self.sent_layernorm_embedding(sent_embeds).squeeze(1)
        x = self.adv(x)
        return x
    