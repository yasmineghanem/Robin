import json


def main():
    with open('./final_intents.json', 'r') as f:
        data = json.load(f)


        intents = data["intents"]

        for intent in intents:
            print(intent)

            corpus = []

            for keyword in intent["keywords"]:
                corpus.append(keyword)


            # add a period at the end of each sentence
            corpus = [sentence + ".\n" for sentence in corpus if sentence[-1] != "."]

            # join the sentences with a space
            corpus = " ".join(corpus)

            file_name = intent["intent"].lower().replace(" ", "_") + ".txt"
            with open(f'./ner_dataset/{file_name}', 'w') as f:
                f.write(corpus)

if __name__ == "__main__":
    print("Preparing NER dataset")
    main()