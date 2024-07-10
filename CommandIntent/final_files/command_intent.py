import sys
from intent_detection_model import load_model, predict_intent
from ner_model import NERModel, predict_entities
from data_preparation import IntentData, NERData
from post_processor import PostProcessor
import torch

class CommandIntent:
    def __init__(self, intent_model_path, ner_model_path):
        self.intent_data = IntentData('../intent_detection_dataset/final_intents_dataset.json')
        self.ner_data = NERData('../ner_dataset/ner_dataset.csv', '../ner_dataset/intent_to_tags.json')
        
        # load the intent model
        self.intent_model = load_model(intent_model_path)
        
        # load the ner model
        self.ner_model = NERModel(vocab_size=self.ner_data.vocab_size, index_to_tag=self.ner_data.index_to_tag)
        state_dict = torch.load(ner_model_path)
        self.ner_model.load_state_dict(state_dict)

        self.post_processor = PostProcessor(self.ner_data.intent_to_tags)

    def __get_intent(self, command):
        '''
            This function takes the command as input and returns the intent of the command.

            Args:
            - command: str: The command given by the user.

            Returns:
            - intent: str: The intent of the command.
        '''
        # prepare the input for the intent model
        intent_model_input = self.intent_data.prepare_input_for_prediction(command)

        # predict the intent of the user input
        intent_prediction = predict_intent(self.intent_model, intent_model_input)

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
        ner_model_input, ner_intent_input, intent_mask = self.ner_data.prepare_input_for_prediction(command, intent)
        
        # predict the entities in the user input
        ner_prediction = predict_entities(self.ner_model, ner_model_input, ner_intent_input, intent_mask)

        # get the entities
        entities = self.ner_data.get_entities(ner_prediction)

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

        # get the entities of the command
        entities = self.__get_entities(command, intent)

        # post process the entities
        response = self.post_processor.post_process(command, intent, entities)


        return response