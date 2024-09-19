"""Start a Flask server to interact with the model.

Inspired by `script/ask.py`."""

import argparse
import json
import logging
import random
import uuid
from typing import Any, Dict, Tuple

import openai
from flask import Flask, request, session

from src.model.crs_model import CRSModel
from src.model.utils import get_entity, get_options

logging.basicConfig(
    format="[%(asctime)s] %(levelname)-12s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command line arguments.

    Returns:
        Command line arguments.
    """
    parser = argparse.ArgumentParser(
        prog="serve_model.py",
        description="Start a Flask server to interact with the model.",
    )

    parser.add_argument(
        "--crs_model",
        type=str,
        choices=["kbrd", "barcor", "unicrs", "chatgpt"],
    )

    parser.add_argument(
        "--kg_dataset", type=str, choices=["redial", "opendialkg"]
    )

    # model_detailed
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--entity_hidden_size", type=int)
    parser.add_argument("--num_bases", type=int, default=8)
    parser.add_argument("--context_max_length", type=int)
    parser.add_argument("--entity_max_length", type=int)

    # model
    parser.add_argument("--rec_model", type=str)
    parser.add_argument("--conv_model", type=str)

    # conv
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--encoder_layers", type=int)
    parser.add_argument("--decoder_layers", type=int)
    parser.add_argument("--text_hidden_size", type=int)
    parser.add_argument("--attn_head", type=int)
    parser.add_argument("--resp_max_length", type=int)

    # prompt
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--text_tokenizer_path", type=str)
    parser.add_argument("--text_encoder", type=str)

    # server
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="5005")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def get_model_args(
    model_name: str, args: argparse.Namespace
) -> Dict[str, Any]:
    """Returns model's arguments from command line arguments.

    Args:
        model_name: Model's name.
        args: Command line arguments.

    Raises:
        ValueError: If model is not supported.

    Returns:
        Model's arguments.
    """
    if model_name == "kbrd":
        return {
            "debug": args.debug,
            "kg_dataset": args.kg_dataset,
            "hidden_size": args.hidden_size,
            "entity_hidden_size": args.entity_hidden_size,
            "num_bases": args.num_bases,
            "rec_model": args.rec_model,
            "conv_model": args.conv_model,
            "context_max_length": args.context_max_length,
            "entity_max_length": args.entity_max_length,
            "tokenizer_path": args.tokenizer_path,
            "encoder_layers": args.encoder_layers,
            "decoder_layers": args.decoder_layers,
            "text_hidden_size": args.text_hidden_size,
            "attn_head": args.attn_head,
            "resp_max_length": args.resp_max_length,
            "seed": args.seed,
        }
    elif model_name == "barcor":
        return {
            "debug": args.debug,
            "kg_dataset": args.kg_dataset,
            "rec_model": args.rec_model,
            "conv_model": args.conv_model,
            "context_max_length": args.context_max_length,
            "resp_max_length": args.resp_max_length,
            "tokenizer_path": args.tokenizer_path,
            "seed": args.seed,
        }
    elif model_name == "unicrs":
        return {
            "debug": args.debug,
            "seed": args.seed,
            "kg_dataset": args.kg_dataset,
            "tokenizer_path": args.tokenizer_path,
            "context_max_length": args.context_max_length,
            "entity_max_length": args.entity_max_length,
            "resp_max_length": args.resp_max_length,
            "text_tokenizer_path": args.text_tokenizer_path,
            "rec_model": args.rec_model,
            "conv_model": args.conv_model,
            "model": args.model,
            "num_bases": args.num_bases,
            "text_encoder": args.text_encoder,
        }
    elif model_name == "chatgpt":
        openai.api_key = args.api_key
        return {
            "seed": args.seed,
            "debug": args.debug,
            "kg_dataset": args.kg_dataset,
        }

    raise ValueError(f"Model {model_name} is not supported.")


class CRSFlaskServer:
    def __init__(
        self,
        crs_model: CRSModel,
        kg_dataset: str,
        response_generation_args: Dict[str, Any] = {},
    ) -> None:
        """Initializes CRS Flask server.

        Args:
            crs_model: CRS model.
            kg_dataset: Name of knowledge graph dataset.
            response_generation_args: Arguments for response generation.
              Defaults to None.
        """
        self.crs_model = crs_model

        # Load entity data
        with open(
            f"data/{kg_dataset}/entity2id.json", "r", encoding="utf-8"
        ) as f:
            self.entity2id = json.load(f)

        self.id2entity = {int(v): k for k, v in self.entity2id.items()}
        self.entity_list = list(self.entity2id.keys())

        # Get options
        self.options = get_options(kg_dataset)

        # Response generation arguments
        self.response_generation_args = response_generation_args

        self.app = Flask(__name__)
        self.app.add_url_rule(
            "/",
            "receive_message",
            self.receive_message,
            methods=["GET", "POST"],
        )
        self.app.secret_key = str(uuid.uuid4().hex)

    def start(self, host: str = "127.0.0.1", port: str = "5005") -> None:
        """Starts the CRS Flask server.

        Args:
            host: Host address. Defaults to 127.0.0.1.
            port: Port number. Defaults to 5005.
        """
        self._host = host
        self._port = port
        self.app.run(host=host, port=port)

    def receive_message(self) -> Tuple[Dict[str, Any], int]:
        """Receives a message and returns a response.

        Returns:
            A response dictionary with the message and status code.
        """
        if request.method == "GET":
            return "Model is running.", 200
        else:
            sender_data = request.get_json()
            logger.debug(f"Received user request:\n{sender_data}")

            try:
                # Process conversation to create conversation dictionary
                conversation_dict = self._process_sender_data(sender_data)
                state = conversation_dict.pop("state")

                # Get response
                response, new_state = self.crs_model.get_response(
                    conversation_dict,
                    self.id2entity,
                    self.options,
                    state,
                    **self.response_generation_args,
                )
                logger.debug(f"Generated response: {response}")
                session["state"] = new_state
                return {"response": response}, 200
            except ValueError as e:
                logger.error(f"Error occurred: {e}")
                return (
                    "An error occurred, make sure you have provided the context"
                    " and message.",
                    400,
                )

    def _process_sender_data(
        self, sender_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Processes sender data to create conversation dictionary.

        The conversation dictionary contains the following keys: context,
        entity, rec, resp, template, and state. Context is a list of the
        previous utterances, entity is a list of entities mentioned in the
        conversation, rec is the recommended items, resp is the response
        generated by the model, and state is the state of the options.
        Note that rec, resp, and template are empty as the model is used for
        inference only, they are kept for compatibility with the models.

        Args:
            sender_data: Data sent by the sender.

        Raises:
            ValueError: If context or message is not present in sender data.

        Returns:
            Conversation dictionary.
        """
        if any(key not in sender_data for key in ["context", "message"]):
            raise ValueError(
                "Invalid sender data. Missing context or message."
            )

        context = sender_data["context"] + [sender_data["message"]]

        state = session.pop("state", None)
        if state is None or len(state) != len(self.options[1]):
            state = [0.0] * len(self.options[1])

        entities = []
        for utterance in context:
            utterance_entities = get_entity(utterance, self.entity_list)
            entities.extend(utterance_entities)

        return {
            "context": context,
            "entity": entities,
            "rec": [],
            "resp": "",
            "template": [],
            "state": state,
        }


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    model_args = get_model_args(args.crs_model, args)
    logger.info(f"Loaded arguments for {args.crs_model} model.")
    logger.debug(f"Model arguments:\n{model_args}")

    # Load model
    crs_model = CRSModel(crs_model=args.crs_model, **model_args)
    logger.info(f"Loaded {args.crs_model} model.")

    # Generation arguments
    response_generation_args = {}
    if args.crs_model == "unicrs":
        response_generation_args = {
            "movie_token": "<pad>",
        }

    # Start CRS Flask server
    crs_server = CRSFlaskServer(
        crs_model, args.kg_dataset, response_generation_args
    )
    crs_server.start(args.host, args.port)
