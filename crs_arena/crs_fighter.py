"""CRS Fighter.

This class represents a CRS fighter. A CRS fighter has a fighter id (i.e., 1
or 2), a name (i.e., model name), and a CRS. The CRS is loaded using the
model name and configuration file.
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from utils import get_crs_model

from src.model.utils import get_entity, get_options

if TYPE_CHECKING:
    from battle_manager import Message


class CRSFighter:
    def __init__(self, fighter_id: int, name: str, config_path: str) -> None:
        """Initializes CRS fighter.

        Args:
            fighter_id: Fighter id (1 or 2).
            name: Model name.
            config: Model configuration file.

        Raises:
            ValueError: If id is not 1 or 2.
        """
        if fighter_id not in [1, 2]:
            raise ValueError("Fighter id must be 1 or 2.")

        self.fighter_id = fighter_id

        self.name = name
        self.config_path = config_path
        self.model = get_crs_model(self.name, self.config_path)

        # Load entity data
        self._load_entity_data()

        # Load options
        self.options = get_options(self.model.crs_model.kg_dataset)

        # Generation arguments.
        self.response_generation_args = {}
        if self.name.split("_")[0] == "unicrs":
            self.response_generation_args.update(
                {
                    "movie_token": "<pad>",
                }
            )

    def _load_entity_data(self):
        """Loads entity data."""
        with open(
            f"data/{self.model.crs_model.kg_dataset}/entity2id.json",
            "r",
            encoding="utf-8",
        ) as f:
            self.entity2id = json.load(f)

        self.id2entity = {int(v): k for k, v in self.entity2id.items()}
        self.entity_list = list(self.entity2id.keys())

    def _process_user_input(
        self, input_message: str, history: List["Message"]
    ) -> Dict[str, Any]:
        """Processes user input.

        The conversation dictionary contains the following keys: context,
        entity, rec, and resp. Context is a list of the previous utterances,
        entity is a list of entities mentioned in the conversation, rec is the
        recommended items, resp is the response generated by the model, and
        template is the context with masked entities.
        Note that rec, resp, and template are empty as the model is used for
        inference only, they are kept for compatibility with the models.

        Args:
            input_message: User input message.
            history: Conversation history.

        Returns:
            Processed user input.
        """
        context = [m["message"] for m in history] + [input_message]
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
        }

    def reply(
        self,
        input_message: str,
        history: List["Message"],
        options_state: Optional[List[float]],
    ) -> Tuple[str, List[float]]:
        """Generates a reply to the user input.

        Args:
            input_message: User input message.
            history: Conversation history.
            options_state: State of the options.

        Returns:
            Generated response and updated state.
        """
        # Process conversation to create conversation dictionary
        conversation_dict = self._process_user_input(input_message, history)

        if options_state is None or len(options_state) != len(self.options[1]):
            options_state = [0.0] * len(self.options[1])

        # Get response
        response, state = self.model.get_response(
            conversation_dict,
            self.id2entity,
            self.options,
            options_state,
            **self.response_generation_args,
        )
        return response, state
