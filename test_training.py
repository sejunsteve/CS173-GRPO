"""Offline integration: real TRL sampling, GRPO loss, and optimizer on a tiny random LM.

This is an infrastructure test with a synthetic reward, not access-task evidence.
"""
import tempfile
import unittest
import json
from pathlib import Path


class TrainerIntegrationTests(unittest.TestCase):
    def test_lora_checkpointing_preserves_adapter_gradients(self):
        import torch
        from datasets import Dataset
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast, set_seed
        from peft import LoraConfig, get_peft_model
        from trl import GRPOTrainer
        from training import make_config
        torch.set_num_threads(2)
        set_seed(173)
        raw=Tokenizer(WordLevel({'<pad>':0,'<eos>':1,'<unk>':2,'request':3,'a':4,'b':5,'c':6,'d':7},unk_token='<unk>'))
        raw.pre_tokenizer=Whitespace()
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw,pad_token='<pad>',eos_token='<eos>',unk_token='<unk>')
        tokenizer.padding_side='left'
        tokenizer.model_input_names=['input_ids','attention_mask']
        base=Qwen2ForCausalLM(Qwen2Config(vocab_size=8,hidden_size=24,intermediate_size=48,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,
            max_position_embeddings=32,eos_token_id=1,pad_token_id=0,bos_token_id=3))
        model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
        model.enable_input_require_grads()
        before={name:value.detach().clone() for name,value in model.named_parameters()}
        trainable={name for name,value in model.named_parameters() if value.requires_grad}
        def synthetic_reward(completions,completion_ids,**kwargs):
            return [float(ids[0])/7 if len(ids) else 0.0 for ids in completion_ids]
        with tempfile.TemporaryDirectory() as tmp:
            config=make_config(Path(tmp),steps=2,cpu=True)
            config.max_prompt_length=8
            config.max_completion_length=4
            config.mask_truncated_completions=False
            config.learning_rate=1e-3
            config.save_strategy='no'
            config.disable_tqdm=True
            trainer=GRPOTrainer(model=model,args=config,reward_funcs=synthetic_reward,
                train_dataset=Dataset.from_list([{'prompt':'request'}]*8),processing_class=tokenizer)
            trainer.train()
        self.assertTrue(any(not torch.equal(before[n],p.detach()) for n,p in model.named_parameters() if n in trainable))
        self.assertTrue(all(torch.equal(before[n],p.detach()) for n,p in model.named_parameters() if n not in trainable))

    def test_real_grpo_update_without_network(self):
        import torch
        from datasets import Dataset
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, set_seed
        from trl import GRPOTrainer
        from training import make_config
        torch.set_num_threads(2)
        set_seed(173)
        vocab = {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'request': 3,
                 'ALLOW': 4, 'DENY': 5, 'R01': 6, 'R10': 7, 'x': 8}
        raw = Tokenizer(WordLevel(vocab, unk_token='<unk>'))
        raw.pre_tokenizer = Whitespace()
        tok = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='<pad>', eos_token='<eos>', unk_token='<unk>')
        tok.padding_side = 'left'
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_embd=24, n_layer=1, n_head=2,
                                         n_positions=32, bos_token_id=3, eos_token_id=1, pad_token_id=0,
                                         resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
        seen = []
        def synthetic_reward(completions, completion_ids, request_json, expected_json, **kwargs):
            from rewards import access_reward
            self.assertEqual(len(completions), len(request_json))
            self.assertEqual(access_reward(expected_json, request_json), [1.0] * len(completions))
            for start in range(0,len(request_json),4):
                self.assertEqual(len(set(request_json[start:start+4])),1)
            seen.extend(completion_ids)
            return [float(ids[0]) / 8 if len(ids) else 0.0 for ids in completion_ids]
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / 'random-model'
            model.save_pretrained(model_dir)
            model.config._name_or_path = str(model_dir)
            config = make_config(Path(tmp), steps=2, cpu=True)
            config.max_prompt_length = 8
            config.max_completion_length = 4
            config.gradient_checkpointing = False
            config.mask_truncated_completions = False
            config.learning_rate = 1e-3
            config.save_strategy = 'no'
            config.disable_tqdm = True
            from test_contract import BASE
            data=[]
            for index in range(8):
                request=dict(BASE,resource_state='locked' if index%2 else 'active')
                expected={'decision':'DENY','rule_id':'R01'} if index%2 else {'decision':'ALLOW','rule_id':'R10'}
                data.append({'prompt':'request','request_json':json.dumps(request),'expected_json':json.dumps(expected)})
            trainer = GRPOTrainer(model=model, args=config, reward_funcs=synthetic_reward,
                                  train_dataset=Dataset.from_list(data), processing_class=tok)
            before = {name: value.detach().clone() for name, value in model.named_parameters()}
            trainer.train()
            self.assertEqual(trainer.state.global_step, 2)
            self.assertGreaterEqual(len(seen), config.num_generations)
            self.assertTrue(any(not torch.equal(before[name], value.detach()) for name, value in model.named_parameters()))
            self.assertTrue(all(torch.isfinite(value).all() for value in model.parameters()))


if __name__ == '__main__':
    unittest.main()
