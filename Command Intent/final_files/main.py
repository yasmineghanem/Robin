'''
This is the main file that will be executed when the user runs the program.
Steps:
1. laod the models
2. get the user input (speech recognition module)
3. predict the intent of the user input
4. pass the user input along with the intent to the ner model
5. pass the output of the ner model along with the intent to the post processor
'''

import sys
from intent_detection_model import load_model, predict_intent
from ner_model import NERModel, predict_entities
from data_preparation import IntentData, NERData
from post_processor import PostProcessor
import torch


def main():
    # for now the user input is hardcoded
    input = "list all the files in the program"

    interactive = True
    mouse = False

    # read the saved models path from the command line arguments
    # remove the name of the script
    args = sys.argv[1:]

    # check that the path for both models are provided
    if len(args) != 2:
        print("Usage: python main.py <intent_model_path> <ner_model_path>")
        sys.exit(1)

    # declarations
    intent_data = IntentData(
        '../intent_detection_dataset/final_intents_dataset.json')
    ner_data = NERData('../ner_dataset/ner_dataset.csv',
                       '../ner_dataset/intent_to_tags.json')

    # load models from the provided paths
    # 1. load the intent model
    intent_model_path = args[0]
    intent_model = load_model(intent_model_path)

    intent_model_input = intent_data.prepare_input_for_prediction(input)

    # predict the intent of the user input
    intent_prediction = predict_intent(intent_model, intent_model_input)

    # get the intent
    intent = intent_data.get_intent(intent_prediction).lower()

    print("Intent:", intent)

    ner_model_path = args[1]

    ner_model = NERModel(vocab_size=ner_data.vocab_size,
                         index_to_tag=ner_data.index_to_tag)
    state_dict = torch.load(ner_model_path)
    ner_model.load_state_dict(state_dict)
    # 2. load the ner model

    if 'activate' in intent:
        print("Activating...")
        if 'interactive' in intent:
            interactive = True
        else:
            mouse = True
    else:
        if intent == 'interactive commands':
            if not interactive:
                print("Interactive commands can only be used in interactive mode")
                sys.exit(1)

        ner_model_input, ner_intent_input, intent_mask = ner_data.prepare_input_for_prediction(
            input, intent)

        # predict the entities in the user input
        ner_prediction = predict_entities(
            ner_model, ner_model_input, ner_intent_input, intent_mask)

        entities = ner_data.get_entities(ner_prediction)

        print("Entities: ", entities)

        post_processor = PostProcessor(ner_data.intent_to_tags)

        parameters = post_processor.post_process(input, intent, entities)
        print("Final parameters to send: ", parameters)


if __name__ == '__main__':
    main()
