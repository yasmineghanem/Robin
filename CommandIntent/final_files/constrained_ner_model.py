import torch
import torch.nn as nn
from TorchCRF import CRF
from typing import List, Tuple, Optional
import tqdm

INF = -100


class ConstrainedCRF(CRF):
    def __init__(self, num_tags):
        super(ConstrainedCRF, self).__init__(num_tags)

    def decode(self, emissions: torch.Tensor, constraints: List[Tuple[torch.IntTensor, torch.IntTensor]], mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions:  
            mask:  
            constraints: List of tuples containing the constraints for the CRF.

        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask, constraints)

    def _viterbi_decode(self, emissions: torch.FloatTensor, mask: torch.ByteTensor, constraints: List[Tuple[torch.IntTensor, torch.IntTensor]]) -> List[List[int]]:
        print("Viterbi Decode")
        '''
            override the viterbi decode function to include the constraints
        '''
        assert emissions.dim() == 3 and mask.dim(
        ) == 2  # (sequence_length, batch_size, num_tags)
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        # get the sequence length and batch size
        sequence_length, batch_size = mask.shape

        # first we apply the constraints to the transitions
        constrained_transitions = self.transitions.clone()
        constrainted_satrt_transitions = self.start_transitions.clone()
        for constraint in constraints:
            if constraint[0] == -1:
                # start transition constraint
                constrainted_satrt_transitions[constraint[1]] = INF
            else:
                constrained_transitions[constraint[0], constraint[1]] = INF

        # start transition and first emission
        # tensor of size (batch_size, num_tags)
        # constraints for start transition
        score = constrainted_satrt_transitions + emissions[0]

        # back pointers to store the tag sequences that give the best score
        # keeps tracks of the best paths up to the current timestep
        backpointers = []

        # we start from 1 since the first timestep is already computed
        # using the start transition and the first emission
        for i in range(1, sequence_length):

            # broadcast the current score for every possible next tag
            # the score contains the score of the best path up to the current timestep
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # emission[i] is the emission score for the current timestep
            # broadcast the emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # the addition of the broadcasted score and the broadcasted emission and the trainsition score
            # result int the shape (batch_size, num_tags, num_tags) where each entry (i, j, k)
            # the score for transitioning from tag j to tag k at the current timestep
            # and emitting k
            next_score = broadcast_score + constrained_transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags) the maximum score for each possible current tag to update the score
            # indices: (batch_size, num_tags) the index of the tag that gives the maximum score
            next_score, indices = next_score.max(dim=1)

            # update the score and the backpointers
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

            # add the indices to the backpointers
            backpointers.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now that we have the final score, we can trace back the best path

        # shape: (batch_size,) containing the length of each sequence
        seq_ends = mask.long().sum(dim=0) - 1

        # shape: (batch_size, num_tags) containing the score of the best tag for each sequence
        # in our case will only have one since our batch size is 1
        best_tags_list = []

        # loop on tags for each sequence in the batch (again 1)
        for idx in range(batch_size):

            # find the best tag for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # backtrace the best tag for each timestep
            # we move from the last timestep to the first
            for hist in reversed(backpointers[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # reverse the order
            best_tags.reverse()

            # again we only have one sequence in the batch
            best_tags_list.append(best_tags)

        return best_tags_list


class NERModel(nn.Module):
    def __init__(self, vocab_size, word_embedding_dim, intent_embedding_dim, hidden_dim, output_dim, number_of_intents, index_to_tag):

        super(NERModel, self).__init__()
        # hyperparameters
        self.word_embedding_dim = word_embedding_dim
        self.inten_embedding_dim = intent_embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.index_to_tag = index_to_tag

        # model layers
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim)
        self.intent_embedding = nn.Embedding(
            number_of_intents, intent_embedding_dim)
        self.lstm = nn.LSTM(word_embedding_dim + intent_embedding_dim,
                            hidden_dim // 2, bidirectional=True, dropout=0.4)
        self.hidden_to_tag = nn.Linear(hidden_dim, output_dim)
        self.crf = ConstrainedCRF(output_dim)

    def __create_constraints(self, mask, intent):
        constraints = []
        one_indices = torch.where(mask == 1)[0]  # tags that are for the intent
        # tags that are not for the intent
        zero_indices = torch.where(mask == 0)[0]

        print(one_indices)
        print(zero_indices)

        for i in one_indices:
            # add constraints for the tags that are related to the intent
            # the transition from an intent tag to a non-intent tag is not allowed
            for j in zero_indices:
                constraints.append((i, j))
                constraints.append((j, i))

            # constraints that the tag if O can't be followed by I
            # but I can be followed by O
            if self.index_to_tag[i.item()] == 'O':
                for j in one_indices:
                    if self.index_to_tag[j.item()][0] == 'I':
                        constraints.append((i, j))

            # add constraints that if the tag begins with B then the next tag can't begin with B of the same
            # if tag begins with B then it can't be followed by B of the same entity
            # or I of a different entity
            # it can be followed by I of same entity, O or B of a different entity
            # add constraints that if the tag begins with B then the next tag cant begin with I of a different entity
            if self.index_to_tag[i.item()][0] == 'B':
                constraints.append((i, i))
                for j in one_indices:
                    if self.index_to_tag[j.item()][0] == 'I' and self.index_to_tag[j.item()][2:] != self.index_to_tag[i.item()][2:]:
                        constraints.append((i, j))

            # create start constraints
            # if the tag starts with I then it can't be the first tag
            if self.index_to_tag[i.item()][0] == 'I':
                constraints.append((-1, i))
                for j in one_indices:
                    if self.index_to_tag[j.item()][2:] != self.index_to_tag[i.item()][2:]:
                        constraints.append((i, j))
                        constraints.append((j, i))

        # add constraints for the tags that are not related to the intent at all
        for i in zero_indices:
            for j in zero_indices:
                constraints.append((i, j))

        return constraints

    def __init_hidden(self):
        '''
            Initialize the hidden state of the LSTM
        '''
        return (torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2))

    def __get_emissions(self, sentence, intent):
        # initialize the hidden state for the lstm
        # the hidden state is initialized for each input sequence
        # to prevent cascading information from on input sequence to the other
        # as the sentences are independent
        # this also helps prevent overfitting of the model
        self.hidden = self.__init_hidden()

        # generate the word embedding for each word in the input sequence
        # captures the semantic information of the sentence
        word_embeddings = self.word_embedding(
            sentence).view(len(sentence), 1, -1)

        # generate the intent embedding for the sequence intent
        # to sway the model to choose from certain tags based on the inten
        # by learning semantic information about the intent
        # since there is a single intent for each sentence but we concatenate it with the word embedding
        # we repeat the embedding with the length of the sequence
        intent_embeddings = self.intent_embedding(
            intent).view(1, 1, -1).repeat(len(sentence), 1, 1)

        # the final embeddings are concatenated to be passed to the lstm layer
        embeddings = torch.cat((word_embeddings, intent_embeddings), dim=2)

        #
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)

        lstm_features = self.hidden_to_tag(lstm_out)

        return lstm_features

    def negative_log_likelihood(self, sentence, tags, intent, mask):

        emissions = self.__get_emissions(sentence, intent)

        intent_mask = torch.where(
            mask == 0, torch.tensor([INF]), torch.tensor([0.0]))

        emissions = emissions + intent_mask

        tags = tags.view(-1, 1)
        loss = -self.crf(emissions, tags)
        return loss

    def forward(self, sentence, intent, mask):
        # get the emission scores from the BiLSTM
        # the emission score represent the score that a word is a certain tag
        # the score is calculated based on the word embedding and the intent embedding
        emissions = self.__get_emissions(sentence, intent)

        # since we are using the negative log likelihood loss
        # map the 0s to -inf and the 1s to 0
        # as the scores could be are negative
        intent_mask = torch.where(
            mask == 0, torch.tensor([INF]), torch.tensor([0.0]))

        # adjust the emissions to account for the intent mask
        # the emission score for the tags that are not related to the intent are set to -inf
        emissions = emissions + intent_mask

        # baseed on the mask and intent create the constraints
        # the constraints are used to guide the CRF to select the best tag sequence
        # by adjusting the transition scores from one tag to another
        constraints = self.__create_constraints(mask, intent)

        # decode the best tag sequence using the CRF
        # using the emissions and the constraints
        tag_sequence = self.crf.decode(emissions, constraints)

        return tag_sequence

# training loop


def create_mask(intent, tags, intent_to_tags):
    pass


def prepare_sequence(input_sequnece, word_to_index):
    pass


def train(model, optimizer, training_data, all_tags, intent_to_tags, word_to_index, intent_to_index, tag_to_index, epochs=10):
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for sentence, tags, intent in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            intent_mask = create_mask(intent, all_tags, intent_to_tags)

            sentence = prepare_sequence(sentence.split(), word_to_index)
            target_tags = torch.tensor([tag_to_index[t]
                                       for t in tags], dtype=torch.long)
            intent = torch.tensor([intent_to_index[intent]], dtype=torch.long)

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(
                sentence, target_tags, intent, intent_mask)

            total_loss += loss.item()

            # Step 4. Compute the loss, gradients, and update the parameters by
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch}, Loss: {total_loss / len(training_data)}")


def evaluate():
    pass


def load_ner_model(path):
    return torch.load(path)


def save_ner_model(model, path):
    torch.save(model, path)


def predict_entities(model, input_sentence, intent, intent_mask):
    model.eval()
    with torch.no_grad():
        entities = model(input_sentence, intent, intent_mask)[0]

    return entities
