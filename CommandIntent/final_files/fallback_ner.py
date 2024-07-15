import json
import os
import re


class FallbackNER():
    def __init__(self, templates_path):
        self.templates_path = os.path.abspath(templates_path)

        self.intent_templates = self.__load_templates()

    def __load_templates(self):
        with open(self.templates_path, 'r') as templates_file:
            templates = json.load(templates_file)

        return templates

    def get_entities(self, command, intent):
        '''
            This function takes the command and intent as input and returns the entities extracted from the command.

            Args:
            - command: str: The command given by the user.
            - intent: str: The intent of the command.

            Returns:
            - entities: list: The entities extracted from the command.
        '''
        print('inside fallback ner')
        # print(intent.capitalize())
        print(intent)
        for pattern in self.intent_templates[intent]:
            # Case-insensitive matching
            # print(pattern)
            match = re.match(pattern, command, re.IGNORECASE)
            if match:
                print('matched')
                entities = match.groupdict()
                print(entities)
                return entities
        print('not matched')
        return {}
