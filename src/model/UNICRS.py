import json
import logging
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoModel, AutoTokenizer

sys.path.append("..")

from src.model.unicrs.config import get_special_tokens_dict
from src.model.unicrs.kg_unicrs import KGForUniCRS
from src.model.unicrs.model_gpt2 import PromptGPT2forCRS
from src.model.unicrs.model_prompt import KGPrompt
from src.model.utils import padded_tensor


class UNICRS:
    def __init__(
        self,
        seed,
        kg_dataset,
        debug,
        tokenizer_path,
        context_max_length,
        entity_max_length,
        resp_max_length,
        text_tokenizer_path,
        model,
        text_encoder,
        num_bases,
        rec_model,
        conv_model,
    ):
        if seed is not None:
            set_seed(seed)

        self.debug = debug

        self.accelerator = Accelerator(
            device_placement=False, mixed_precision="fp16"
        )
        self.device = self.accelerator.device

        self.context_max_length = context_max_length
        self.entity_max_length = entity_max_length
        self.resp_max_length = resp_max_length

        self.padding = "max_length"
        self.pad_to_multiple_of = 8

        self.tokenizer_path = tokenizer_path
        self.text_tokenizer_path = text_tokenizer_path

        self.text_encoder = text_encoder
        self.model_path = model
        self.rec_model_path = rec_model
        self.conv_model_path = conv_model

        # config
        gpt2_special_tokens_dict, prompt_special_tokens_dict = (
            get_special_tokens_dict(kg_dataset)
        )

        # backbone
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.tokenizer.add_special_tokens(gpt2_special_tokens_dict)
        self.tokenizer.padding_side = "left"

        self.model = PromptGPT2forCRS.from_pretrained(self.model_path)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = self.model.to(self.device)

        # text prompt encoder
        self.prompt_tokenizer = AutoTokenizer.from_pretrained(
            self.text_tokenizer_path
        )
        self.prompt_tokenizer.add_special_tokens(prompt_special_tokens_dict)

        self.text_encoder = AutoModel.from_pretrained(self.text_encoder)
        self.text_encoder.resize_token_embeddings(len(self.prompt_tokenizer))
        self.text_encoder = self.text_encoder.to(self.device)

        # kg prompt
        self.kg_dataset = kg_dataset
        self.kg = KGForUniCRS(
            kg=self.kg_dataset, debug=self.debug
        ).get_kg_info()
        self.item_ids = torch.as_tensor(
            self.kg["item_ids"], device=self.device
        )
        self.kg_dataset_path = f"data/{self.kg_dataset}"
        with open(
            f"{self.kg_dataset_path}/entity2id.json", "r", encoding="utf-8"
        ) as f:
            self.entity2id = json.load(f)
        self.entity_pad_id = self.kg["pad_entity_id"]

        self.num_bases = num_bases
        # prompt for rec
        self.rec_prompt_encoder = KGPrompt(
            self.model.config.n_embd,
            self.text_encoder.config.hidden_size,
            self.model.config.n_head,
            self.model.config.n_layer,
            2,
            n_entity=self.kg["num_entities"],
            num_relations=self.kg["num_relations"],
            num_bases=self.num_bases,
            edge_index=self.kg["edge_index"],
            edge_type=self.kg["edge_type"],
        )
        if rec_model is not None:
            self.rec_prompt_encoder.load(self.rec_model_path)
        self.rec_prompt_encoder = self.rec_prompt_encoder.to(self.device)
        self.rec_prompt_encoder = self.accelerator.prepare(
            self.rec_prompt_encoder
        )

        # prompt for conv
        self.conv_prompt_encoder = KGPrompt(
            self.model.config.n_embd,
            self.text_encoder.config.hidden_size,
            self.model.config.n_head,
            self.model.config.n_layer,
            2,
            n_entity=self.kg["num_entities"],
            num_relations=self.kg["num_relations"],
            num_bases=self.num_bases,
            edge_index=self.kg["edge_index"],
            edge_type=self.kg["edge_type"],
        )
        if conv_model is not None:
            self.conv_prompt_encoder.load(self.conv_model_path)
        self.conv_prompt_encoder = self.conv_prompt_encoder.to(self.device)
        self.conv_prompt_encoder = self.accelerator.prepare(
            self.conv_prompt_encoder
        )

    def get_rec(self, conv_dict):
        text_list = []
        turn_idx = 0
        for utt in conv_dict["context"]:
            if utt != "":
                text = ""
                if turn_idx % 2 == 0:
                    text += "User: "
                else:
                    text += "System: "
                text += utt
                text_list.append(text)
            turn_idx += 1
        context = f"{self.tokenizer.eos_token}".join(text_list)
        context += f"{self.tokenizer.eos_token}"
        prompt_context = f"{self.prompt_tokenizer.sep_token}".join(text_list)

        self.tokenizer.truncation_side = "left"
        context_ids = self.tokenizer.encode(
            context, truncation=True, max_length=self.context_max_length
        )

        self.prompt_tokenizer.truncation_side = "left"
        prompt_ids = self.prompt_tokenizer.encode(
            prompt_context, truncation=True, max_length=self.context_max_length
        )

        self.data_list = []

        if "rec" not in conv_dict.keys() or not conv_dict["rec"]:
            # Interactive mode: the ground truth is not provided
            data_dict = {
                "context": context_ids,
                "prompt": prompt_ids,
                "entity": [
                    self.entity2id[ent]
                    for ent in conv_dict["entity"][-self.entity_max_length :]
                    if ent in self.entity2id
                ],
            }
            self.data_list.append(data_dict)
        else:
            for rec in conv_dict["rec"]:
                if rec in self.entity2id:
                    data_dict = {
                        "context": context_ids,
                        "prompt": prompt_ids,
                        "entity": [
                            self.entity2id[ent]
                            for ent in conv_dict["entity"][
                                -self.entity_max_length :
                            ]
                            if ent in self.entity2id
                        ],
                        "rec": self.entity2id[rec],
                    }
                    self.data_list.append(data_dict)

        context_dict = defaultdict(list)
        prompt_dict = defaultdict(list)
        entity_list = []
        label_list = []

        for data in self.data_list:
            context_dict["input_ids"].append(data["context"])
            prompt_dict["input_ids"].append(data["prompt"])
            entity_list.append(data["entity"])
            if "rec" in data.keys():
                label_list.append(data["rec"])

        context_dict = self.tokenizer.pad(
            context_dict,
            max_length=self.context_max_length,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        if len(label_list) > 0:
            context_dict["rec_labels"] = label_list

        for k, v in context_dict.items():
            if not isinstance(v, torch.Tensor):
                context_dict[k] = torch.as_tensor(v, device=self.device)

        position_ids = context_dict["attention_mask"].long().cumsum(-1) - 1
        position_ids.masked_fill_(context_dict["attention_mask"] == 0, 1)
        context_dict["position_ids"] = position_ids

        input_batch = {}  # for model
        input_batch["context"] = context_dict

        prompt_dict = self.prompt_tokenizer.pad(
            prompt_dict,
            max_length=self.context_max_length,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        for k, v in prompt_dict.items():
            if not isinstance(v, torch.Tensor):
                prompt_dict[k] = torch.as_tensor(v, device=self.device)
        input_batch["prompt"] = prompt_dict

        entity_list = padded_tensor(
            entity_list,
            pad_id=self.entity_pad_id,
            pad_tail=True,
            device=self.device,
            debug=self.debug,
            max_length=self.entity_max_length,
        )
        input_batch["entity"] = entity_list

        # infer
        token_embeds = self.text_encoder(
            **input_batch["prompt"]
        ).last_hidden_state
        prompt_embeds = self.rec_prompt_encoder(
            entity_ids=input_batch["entity"],
            token_embeds=token_embeds,
            output_entity=True,
        )
        input_batch["context"]["prompt_embeds"] = prompt_embeds
        input_batch["context"][
            "entity_embeds"
        ] = self.rec_prompt_encoder.get_entity_embeds()

        outputs = self.model(**input_batch["context"], rec=True)
        logits = outputs.rec_logits[:, self.item_ids]
        ranks = torch.topk(logits, k=50, dim=-1).indices
        preds = self.item_ids[ranks].tolist()

        if "rec_labels" in input_batch["context"]:
            labels = input_batch["context"]["rec_labels"].tolist()
        else:
            labels = None

        return preds, labels

    def get_conv(self, conv_dict):
        # dataset

        text_list = []
        turn_idx = 0
        for utt in conv_dict["context"]:
            if utt != "" and len(utt) > 0:
                text = ""
                if turn_idx % 2 == 0:
                    text += "User: "
                else:
                    text += "System: "
                text += utt
                text_list.append(text)
            turn_idx += 1

        context = f"{self.tokenizer.eos_token}".join(text_list)
        context += f"{self.tokenizer.eos_token}"
        prompt_context = f"{self.prompt_tokenizer.sep_token}".join(text_list)

        self.tokenizer.truncation_side = "left"
        context_ids = self.tokenizer.encode(
            context, truncation=True, max_length=self.context_max_length
        )

        self.prompt_tokenizer.truncation_side = "left"
        prompt_ids = self.prompt_tokenizer.encode(
            prompt_context, truncation=True, max_length=self.context_max_length
        )

        self.tokenizer.truncation_side = "right"
        if turn_idx % 2 == 0:
            user_str = "User: "
        else:
            user_str = "System: "
        resp = user_str + conv_dict["resp"]
        resp_ids = self.tokenizer.encode(
            resp, truncation=True, max_length=self.resp_max_length
        )
        resp_ids.append(self.tokenizer.eos_token_id)

        entity_list = [
            self.entity2id[ent]
            for ent in conv_dict["entity"][-self.entity_max_length :]
            if ent in self.entity2id
        ]

        data_dict = {
            "context": context_ids,
            "prompt": prompt_ids,
            "entity": entity_list,
        }

        # dataloader

        context_dict = defaultdict(list)
        context_len_list = []
        prompt_dict = defaultdict(list)
        entity_list = []
        label_dict = defaultdict(list)

        bot_prompt = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize("System:")
        )

        context = data_dict["context"] + bot_prompt
        context_len_list.append((len(data_dict["context"])))
        context_dict["input_ids"] = context

        prompt_dict["input_ids"] = data_dict["prompt"]
        entity_list.append(data_dict["entity"])

        context_max_length = self.context_max_length + len(bot_prompt)

        context_dict = self.tokenizer.pad(
            context_dict,
            max_length=context_max_length,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        for k, v in context_dict.items():
            if not isinstance(v, torch.Tensor):
                context_dict[k] = torch.as_tensor(
                    v, device=self.device
                ).unsqueeze(0)

        input_batch = {}

        position_ids = context_dict["attention_mask"].long().cumsum(-1) - 1
        position_ids.masked_fill_(context_dict["attention_mask"] == 0, 1)
        context_dict["position_ids"] = position_ids

        input_batch["conv_labels"] = label_dict["input_ids"]
        input_batch["context_len"] = context_len_list

        input_batch["context"] = context_dict

        prompt_dict = self.prompt_tokenizer.pad(
            prompt_dict,
            max_length=self.context_max_length,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        for k, v in prompt_dict.items():
            if not isinstance(v, torch.Tensor):
                prompt_dict[k] = torch.as_tensor(
                    v, device=self.device
                ).unsqueeze(0)
        input_batch["prompt"] = prompt_dict

        entity_list = padded_tensor(
            entity_list,
            pad_id=self.entity_pad_id,
            pad_tail=True,
            device=self.device,
            debug=self.debug,
            max_length=self.entity_max_length,
        )
        input_batch["entity"] = entity_list

        # infer

        self.conv_prompt_encoder.eval()

        token_embeds = self.text_encoder(
            **input_batch["prompt"]
        ).last_hidden_state
        prompt_embeds = self.conv_prompt_encoder(
            entity_ids=input_batch["entity"],
            token_embeds=token_embeds,
            output_entity=False,
            use_conv_prefix=True,
        )
        input_batch["context"]["prompt_embeds"] = prompt_embeds

        gen_args = {
            "max_new_tokens": self.resp_max_length,
            "no_repeat_ngram_size": 3,
        }

        gen_seqs = self.model.generate(**input_batch["context"], **gen_args)
        gen_str = self.tokenizer.decode(gen_seqs[0], skip_special_tokens=False)

        return input_batch, gen_str

    def get_choice(self, gen_inputs, options, state, conv_dict=None):
        state = torch.as_tensor(state, device=self.device)
        outputs = self.accelerator.unwrap_model(self.model).generate(
            **gen_inputs["context"],
            min_new_tokens=1,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
        )
        option_token_ids = [
            self.tokenizer.encode(op, add_special_tokens=False)[0]
            for op in options
        ]
        option_scores = outputs.scores[-1][0][option_token_ids]
        option_scores += state
        option_with_max_score = options[torch.argmax(option_scores)]

        return option_with_max_score

    def get_response(
        self,
        conv_dict: Dict[str, Any],
        id2entity: Dict[int, str],
        options: Tuple[str, Dict[str, str]],
        state: List[float],
        movie_token: str = "<pad>",
    ) -> Tuple[str, List[float]]:
        """Generates a response given a conversation context.

        The method is based on the logic of the ask mode (i.e., see
        `scripts/ask.py`). It consists of two steps: (1) choose to either
        recommend items or generate a response, and (2) execute the chosen
        step. Slightly deviates from the original implementation by not using
        templates.

        Args:
            conv_dict: Conversation context.
            id2entity: Mapping from entity ID to entity name.
            options: Prompt with options and dictionary of options.
            state: State of the option choices.
            movie_token: Mask token for the movie. Defaults to "<pad>".

        Returns:
            Generated response and updated state.
        """
        generated_inputs, generated_response = self.get_conv(conv_dict)
        options_letter = list(options[1].keys())

        # Get the choice between recommend and generate
        choice = self.get_choice(generated_inputs, options_letter, state)

        # Generate recommendations
        recommended_items, _ = self.get_rec(conv_dict)

        if choice == options_letter[-1]:
            recommended_items_str = ""
            for i, item_id in enumerate(recommended_items[0][:3]):
                recommended_items_str += f"{i+1}: {id2entity[item_id]}  \n"
            response = (
                "I would recommend the following items:  \n"
                f"{recommended_items_str}"
            )
        else:
            # Original : Generate a response to ask for preferences. The
            # fallback is to use the generated response.
            # response = (
            #     options[1].get(choice, {}).get("template", generated_response)
            # )
            generated_response = generated_response[
                generated_response.rfind("System:") + len("System:") + 1 :
            ]
            generated_response = generated_response.replace(
                "<|endoftext|>", ""
            )

            for i in range(str.count(generated_response, movie_token)):
                try:
                    generated_response = generated_response.replace(
                        movie_token, id2entity[recommended_items[0][i]], 1
                    )
                except IndexError as e:
                    logging.error(e)
                    generated_response = generated_response.replace(
                        movie_token, "", 1
                    )
            response = generated_response.strip()

        # Update the state. Hack: penalize the choice to reduce the
        # likelihood of selecting the same choice again
        state[options_letter.index(choice)] += -1e5

        return response, state
