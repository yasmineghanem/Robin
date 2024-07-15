import sys
import os
from intent_detection_model import load_model, predict_intent
# from ner_model import NERModel, predict_entities
from constrained_ner_model import NERModel, predict_entities
from data_preparation import IntentData, NERData
from post_processor import PostProcessor
import torch
import command_constants
from fallback_ner import FallbackNER


class CommandIntent:
    def __init__(self, intent_model_path, ner_model_path):
        self.intent_model_path = os.path.abspath(intent_model_path)
        self.ner_model_path = os.path.abspath(ner_model_path)
        self.intent_to_tags_path = os.path.abspath(
            '../ner_dataset/intent_to_tags.json')
        self.intent_data = IntentData(self._resolve_path(
            '../intent_detection_dataset/final_intents_dataset.json'))
        self.ner_data = NERData(self._resolve_path(
            '../ner_dataset/ner_dataset.csv'), self._resolve_path('../ner_dataset/intent_to_tags.json'))

        # load the intent model
        self.intent_model = load_model(self.intent_model_path)

        # load the ner model
        self.ner_model = NERModel(vocab_size=1034, word_embedding_dim=50, intent_embedding_dim=50, hidden_dim=64, output_dim=len(
            self.ner_data.tag_to_index), number_of_intents=len(self.ner_data.intent_to_index), index_to_tag=self.ner_data.index_to_tag)
        state_dict = torch.load(self.ner_model_path)
        self.ner_model.load_state_dict(state_dict, strict=False)

        self.fallback_ner = FallbackNER(
            self._resolve_path('../ner_dataset/fallback_ner.json'))

        # self.classical_intent = ClassicalIntent()

        self.post_processor = PostProcessor(self.ner_data.intent_to_tags)

    def _resolve_path(self, relative_path):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

    def __get_intent(self, command):
        '''
            This function takes the command as input and returns the intent of the command.

            Args:
            - command: str: The command given by the user.

            Returns:
            - intent: str: The intent of the command.
        '''
        # prepare the input for the intent model
        intent_model_input = self.intent_data.prepare_input_for_prediction(
            command)

        # predict the intent of the user input
        intent_prediction = predict_intent(
            self.intent_model, intent_model_input)

        # get the intent
        intent = self.intent_data.get_intent(intent_prediction).lower()

        return intent

    def __get_entities(self, command, intent):
        '''
            This function takes the command and intent as input and returns the entities extracted from the command.

            Args:
            - command: str: The command given by the user.
            - intent: str: The intent of the command.

            Returns:
            - entities: list: The entities extracted from the command.
        '''
        # prepare the input for the ner model
        print('inside get entities')
        ner_model_input, ner_intent_input, intent_mask = self.ner_data.prepare_input_for_prediction(
            command, intent)

        print("error hena")
        # predict the entities in the user input
        ner_prediction = predict_entities(
            self.ner_model, ner_model_input, ner_intent_input, intent_mask)

        print("tab error hena")
        # get the entities
        entities = self.ner_data.get_entities(ner_prediction)

        print("errorrr")

        return entities

    def process_command(self, command):
        '''
            This function takes the command as input and returns the parameters extracted from the command.
            to send to the command execution model.

            Args:
            - command: str: The command given by the user.

            Returns:
            - parameters: dict: The parameters extracted from the command.
        '''
        # get the intent of the command
        intent = self.__get_intent(command)
        print(intent)

        fallback = False

        if intent in command_constants.FALLBACK_INTENTS:
            entities = self.fallback_ner.get_entities(command, intent)
            response = self.post_processor.post_process(
                command, intent, entities, True)
            return intent, response

        # get the entities of the command if the intent is not in the no entities list
        # if intent not in command_constants.NO_ENTITIES:
        print('ahlan')
        fallback_entities = self.fallback_ner.get_entities(command, intent)
        print("test:", fallback_entities)
        if fallback_entities:
            print(fallback_entities)
            fallback = True
            response = self.post_processor.post_process(
                command, intent, fallback_entities, fallback=fallback)

            return intent, response

        print('ahlan tany')
        if intent in ['mouse click', 'activate mouse', 'activate interactive']:
            return None

        entities = self.__get_entities(command, intent)
        print(entities)

        print(fallback)
        # post process the entities
        response = self.post_processor.post_process(
            command, intent, entities)

        return intent, response

        # return intent
