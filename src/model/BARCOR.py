import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.append("..")

from src.model.barcor.barcor_model import BartForSequenceClassification
from src.model.barcor.kg_bart import KGForBART


class BARCOR:
    def __init__(
        self,
        seed,
        kg_dataset,
        debug,
        tokenizer_path,
        context_max_length,
        rec_model,
        conv_model,
        resp_max_length,
    ):
        self.seed = seed
        if self.seed is not None:
            set_seed(self.seed)
        self.kg_dataset = kg_dataset

        self.debug = debug
        self.tokenizer_path = tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.tokenizer.truncation_side = "left"
        self.context_max_length = context_max_length

        self.padding = "max_length"
        self.pad_to_multiple_of = 8

        self.accelerator = Accelerator(
            device_placement=False, mixed_precision="fp16"
        )
        self.device = self.accelerator.device

        self.rec_model = rec_model
        self.conv_model = conv_model

        # conv
        self.resp_max_length = resp_max_length

        self.kg = KGForBART(
            kg_dataset=self.kg_dataset, debug=self.debug
        ).get_kg_info()

        self.crs_rec_model = BartForSequenceClassification.from_pretrained(
            self.rec_model, num_labels=self.kg["num_entities"]
        ).to(self.device)
        self.crs_conv_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.conv_model
        ).to(self.device)
        self.crs_conv_model = self.accelerator.prepare(self.crs_conv_model)

        self.kg_dataset_path = f"data/{self.kg_dataset}"
        with open(
            f"{self.kg_dataset_path}/entity2id.json", "r", encoding="utf-8"
        ) as f:
            self.entity2id = json.load(f)

    def get_rec(self, conv_dict):
        # dataset
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

        context = f"{self.tokenizer.sep_token}".join(text_list)
        context_ids = self.tokenizer.encode(
            context, truncation=True, max_length=self.context_max_length
        )

        data_list = []

        if "rec" not in conv_dict.keys() or not conv_dict["rec"]:
            # Interactive mode: the ground truth is not provided
            data_dict = {
                "context": context_ids,
                "entity": [
                    self.entity2id[ent]
                    for ent in conv_dict["entity"]
                    if ent in self.entity2id
                ],
            }
            if "template" in conv_dict:
                data_dict["template"] = conv_dict["template"]
            data_list.append(data_dict)
        else:
            for rec in conv_dict["rec"]:
                if rec in self.entity2id:
                    data_dict = {
                        "context": context_ids,
                        "entity": [
                            self.entity2id[ent]
                            for ent in conv_dict["entity"]
                            if ent in self.entity2id
                        ],
                        "rec": self.entity2id[rec],
                    }
                    if "template" in conv_dict:
                        data_dict["template"] = conv_dict["template"]
                    data_list.append(data_dict)

        # dataloader
        input_dict = defaultdict(list)
        label_list = []

        for data in data_list:
            input_dict["input_ids"].append(data["context"])
            if "rec" in data.keys():
                label_list.append(data["rec"])

        input_dict = self.tokenizer.pad(
            input_dict,
            max_length=self.context_max_length,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        if len(label_list) > 0:
            input_dict["labels"] = label_list

        for k, v in input_dict.items():
            if not isinstance(v, torch.Tensor):
                input_dict[k] = torch.as_tensor(v, device=self.device)

        labels = (
            input_dict["labels"].tolist() if "labels" in input_dict else None
        )
        self.crs_rec_model.eval()
        outputs = self.crs_rec_model(**input_dict)
        item_ids = torch.as_tensor(self.kg["item_ids"], device=self.device)
        logits = outputs["logits"][:, item_ids]
        ranks = torch.topk(logits, k=50, dim=-1).indices
        preds = item_ids[ranks].tolist()

        return preds, labels

    def get_conv(self, conv_dict):
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
        context = f"{self.tokenizer.sep_token}".join(text_list)
        context_ids = self.tokenizer.encode(
            context, truncation=True, max_length=self.context_max_length
        )

        if turn_idx % 2 == 0:
            user_str = "User: "
        else:
            user_str = "System: "
        resp = user_str + conv_dict["resp"]
        resp_ids = self.tokenizer.encode(
            resp, truncation=True, max_length=self.resp_max_length
        )

        data_dict = {
            "context": context_ids,
            "resp": resp_ids,
        }

        input_dict = defaultdict(list)
        label_dict = defaultdict(list)

        input_dict["input_ids"] = data_dict["context"]
        label_dict["input_ids"] = data_dict["resp"]

        input_dict = self.tokenizer.pad(
            input_dict,
            max_length=self.context_max_length,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        label_dict = self.tokenizer.pad(
            label_dict,
            max_length=self.context_max_length,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )["input_ids"]

        input_dict["labels"] = label_dict

        for k, v in input_dict.items():
            if not isinstance(v, torch.Tensor):
                input_dict[k] = torch.as_tensor(
                    v, device=self.device
                ).unsqueeze(0)

        self.crs_conv_model.eval()

        gen_args = {
            "min_length": 0,
            "max_length": self.resp_max_length,
            "num_beams": 1,
            "no_repeat_ngram_size": 3,
            "encoder_no_repeat_ngram_size": 3,
        }

        gen_seqs = self.accelerator.unwrap_model(self.crs_conv_model).generate(
            **input_dict, **gen_args
        )
        gen_str = self.tokenizer.decode(gen_seqs[0], skip_special_tokens=True)

        return input_dict, gen_str

    def get_choice(self, gen_inputs, options, state, conv_dict=None):
        outputs = self.accelerator.unwrap_model(self.crs_conv_model).generate(
            **gen_inputs,
            min_new_tokens=5,
            max_new_tokens=5,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
        )
        option_token_ids = [
            self.tokenizer.encode(f" {op}", add_special_tokens=False)[0]
            for op in options
        ]
        option_scores = outputs.scores[-2][0][option_token_ids]
        state = torch.as_tensor(
            state, device=self.device, dtype=option_scores.dtype
        )
        option_scores += state
        option_with_max_score = options[torch.argmax(option_scores)]

        return option_with_max_score

    def get_response(
        self,
        conv_dict: Dict[str, Any],
        id2entity: Dict[int, str],
        options: Tuple[str, Dict[str, str]],
        state: List[float],
    ) -> Tuple[str, List[float]]:
        """Generates a response given a conversation context.

        Args:
            conv_dict: Conversation context.
            id2entity: Mapping from entity id to entity name.
            options: Prompt with options and dictionary of options.
            state: State of the option choices.

        Returns:
            Generated response and updated state.
        """
        generated_inputs, generated_response = self.get_conv(conv_dict)
        options_letter = list(options[1].keys())

        # Get the choice between recommend and generate
        choice = self.get_choice(generated_inputs, options_letter, state)

        if choice == options_letter[-1]:
            # Generate a recommendation
            recommended_items, _ = self.get_rec(conv_dict)
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
            generated_response = generated_response.lstrip("System;:")
            response = generated_response.strip()

        # Update the state. Hack: penalize the choice to reduce the
        # likelihood of selecting the same choice again
        state[options_letter.index(choice)] = -1e5

        return response, state
