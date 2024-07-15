import torch
import torch.nn as nn
from TorchCRF import CRF
from typing import List, Tuple, Optional

INF = -100

# class ConstrainedCRF(CRF):
#     def __init__(self, num_tags):
#         super(ConstrainedCRF, self).__init__(num_tags)

#     def decode(self, emissions: torch.Tensor, constraints: List[Tuple[torch.IntTensor, torch.IntTensor]], mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
#         """Find the most likely tag sequence using Viterbi algorithm.

#         Args:
#             emissions (`~torch.Tensor`): Emission score tensor of size
#                 ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
#                 ``(batch_size, seq_length, num_tags)`` otherwise.
#             mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
#                 if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

#         Returns:
#             List of list containing the best tag sequence for each batch.
#         """
#         self._validate(emissions, mask=mask)
#         if mask is None:
#             mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

#         if self.batch_first:
#             emissions = emissions.transpose(0, 1)
#             mask = mask.transpose(0, 1)

#         return self._viterbi_decode(emissions, mask, constraints)

#     def _viterbi_decode(self, emissions: torch.FloatTensor, mask: torch.ByteTensor, constraints: List[Tuple[torch.IntTensor, torch.IntTensor]]) -> List[List[int]]:
#             print("Viterbi Decode")
#             '''
#                 override the viterbi decode function to include the constraints
#             '''
#             assert emissions.dim() == 3 and mask.dim() == 2 # (sequence_length, batch_size, num_tags)
#             assert emissions.shape[:2] == mask.shape # (sequence_length, batch_size)
#             assert emissions.size(2) == self.num_tags
#             assert mask[0].all()

#             sequence_length, batch_size = mask.shape

#             constrained_transitions = self.transitions.clone()
#             constrainted_satrt_transitions = self.start_transitions.clone()
#             for constraint in constraints:
#                 if constraint[0] == -1:
#                     constrainted_satrt_transitions[constraint[1]] = INF
#                 else:
#                     constrained_transitions[constraint[0], constraint[1]] = INF



#             # start transition and first emission
#             # tensor of size (batch_size, num_tags)
#             # constraints for start transition
#             # score = self.start_transitions + emissions[0] 
#             score = constrainted_satrt_transitions + emissions[0] 

#             backpointers = []

#             for i in range(1, sequence_length):
#                 # Broadcast viterbi score for every possible next tag
#                 # shape: (batch_size, num_tags, 1)
#                 broadcast_score = score.unsqueeze(2)
                
                    
#                 # Broadcast emission score for every possible current tag
#                 # shape: (batch_size, 1, num_tags)
#                 broadcast_emission = emissions[i].unsqueeze(1)

#                 # Compute the score tensor of size (batch_size, num_tags, num_tags) where
#                 # for each sample, entry at row i and column j stores the score of the best
#                 # tag sequence so far that ends with transitioning from tag i to tag j and emitting
#                 # shape: (batch_size, num_tags, num_tags)
#                 # next_score = broadcast_score + self.transitions + broadcast_emission
#                 next_score = broadcast_score + constrained_transitions + broadcast_emission

#                 # print("Next Score Shape:", next_score[0].shape)

#                 # Apply the constraints
#                 # for constraint in constraints:
#                 #     # print("Constraint:", constraint)
#                 #     next_score[0][constraint[0], constraint[1]] = INF

                
#                 # Find the maximum score over all possible current tag
#                 # shape: (batch_size, num_tags)
#                 next_score, indices = next_score.max(dim=1)

#                 # Set score to the next score if this timestep is valid (mask == 1)
#                 # and save the index that produces the next score
#                 # shape: (batch_size, num_tags)
#                 score = torch.where(mask[i].unsqueeze(1), next_score, score)
#                 backpointers.append(indices)
            
            
#             # End transition score
#             # shape: (batch_size, num_tags)
#             score += self.end_transitions


#             # Now, compute the best path for each sample

#             # shape: (batch_size,)
#             seq_ends = mask.long().sum(dim=0) - 1
#             best_tags_list = []

#             for idx in range(batch_size):
#                 # Find the tag which maximizes the score at the last timestep; this is our best tag
#                 # for the last timestep
#                 _, best_last_tag = score[idx].max(dim=0)
#                 best_tags = [best_last_tag.item()]

#                 # We trace back where the best last tag comes from, append that to our best tag
#                 # sequence, and trace it back again, and so on
#                 for hist in reversed(backpointers[:seq_ends[idx]]):
#                     best_last_tag = hist[idx][best_tags[-1]]
#                     best_tags.append(best_last_tag.item())

#                 # Reverse the order because we start from the last timestep
#                 best_tags.reverse()
#                 best_tags_list.append(best_tags)

#             return best_tags_list



# class NERModel(nn.Module):
#     def __init__(self, vocab_size, word_embedding_dim, intent_embedding_dim, hidden_dim, output_dim, number_of_intents, index_to_tag):
        
#         super(NERModel, self).__init__()
#         # hyperparameters
#         self.word_embedding_dim = word_embedding_dim
#         self.inten_embedding_dim = intent_embedding_dim
#         self.hidden_dim = hidden_dim
#         self.vocab_size = vocab_size
#         self.output_dim = output_dim
#         self.index_to_tag = index_to_tag

#         # model layers
#         self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim)
#         self.intent_embedding = nn.Embedding(number_of_intents, intent_embedding_dim)
#         self.lstm = nn.LSTM(word_embedding_dim + intent_embedding_dim, hidden_dim // 2, bidirectional=True, dropout=0.2)
#         self.hidden_to_tag = nn.Linear(hidden_dim, output_dim)
#         self.crf = ConstrainedCRF(output_dim)

    
#     def __create_constraints(self, mask, intent):
#         constraints = []
#         one_indices = torch.where(mask == 1)[0] # tags that are for the intent
#         zero_indices = torch.where(mask == 0)[0] # tags that are not for the intent
        
#         print(one_indices)
#         print(zero_indices)

#         for i in one_indices:
#             # add constraints for the tags that are related to the intent
#             # the transition from an intent tag to a non-intent tag is not allowed
#             for j in zero_indices:
#                 constraints.append((i, j))
#                 constraints.append((j, i))
            
#             # constraints that the tag if O can't be followed by I
#             # but I can be followed by O 
#             if self.index_to_tag[i.item()] == 'O':
#                 for j in one_indices:
#                     if self.index_to_tag[j.item()][0] == 'I':
#                         constraints.append((i, j))
        
#             # add constraints that if the tag begins with B then the next tag can't begin with B of the same 
#             # if tag begins with B then it can't be followed by B of the same entity
#             # or I of a different entity
#             # it can be followed by I of same entity, O or B of a different entity
#             # add constraints that if the tag begins with B then the next tag cant begin with I of a different entity
#             if self.index_to_tag[i.item()][0] == 'B':
#                 constraints.append((i, i))
#                 for j in one_indices:
#                     if self.index_to_tag[j.item()][0] == 'I' and self.index_to_tag[j.item()][2:] != self.index_to_tag[i.item()][2:]:
#                         constraints.append((i, j))
                        
#             # if self.index_to_tag[i.item()][0] == 'B':
#             #     for j in one_indices:
#             #         if self.index_to_tag[j.item()][0] == 'I' and self.index_to_tag[j.item()][2:] != self.index_to_tag[i.item()][2:]:
#             #             constraints.append((i, j))

#             # create start constraints
#             # if the tag starts with I then it can't be the first tag
#             if self.index_to_tag[i.item()][0] == 'I':
#                 constraints.append((-1, i))
#                 for j in one_indices:
#                     if self.index_to_tag[j.item()][2:] != self.index_to_tag[i.item()][2:]:
#                         constraints.append((i, j))
#                         constraints.append((j, i))
                        
#             # add constraint the if a tag starts with I then the previous tag should be the same entity
#             # the for example I-CONDITION cant be followed by I-VAL and vice versa
#             # if self.index_to_tag[i.item()][0] == 'I':
#             #     for j in one_indices:
#             #         if self.index_to_tag[j.item()][2:] != self.index_to_tag[i.item()][2:]:
#             #             constraints.append((i, j))
#             #             constraints.append((j, i))
                        

#         # add constraints for the tags that are not related to the intent at all
#         for i in zero_indices:
#             for j in zero_indices:
#                 constraints.append((i, j))


#         return constraints

#     def __init_hidden(self):
#         return (torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2)) # initialize hidden state

#     def __get_lstm_features(self, sentence, intent):
#         self.hidden = self.__init_hidden()
#         word_embeddings = self.word_embedding(sentence).view(len(sentence), 1, -1)
#         intent_embeddings = self.intent_embedding(intent).view(1, 1, -1).repeat(len(sentence), 1, 1)
#         embeddings = torch.cat((word_embeddings, intent_embeddings), dim=2) 
#         lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
#         lstm_features = self.hidden_to_tag(lstm_out)
#         return lstm_features

#     def neg_log_likelihood(self, sentence, tags, intent, mask):
#         emissions = self.__get_lstm_features(sentence, intent) # (sentence_length, batch_size, number of tags) # emissions where each word in the sequence coressponds to probability for tag
#         intent_mask = torch.where(mask == 0, torch.tensor([INF]), torch.tensor([0.0]))
#         emissions = emissions + intent_mask
#         tags = tags.view(-1, 1)
#         loss = -self.crf(emissions, tags)
#         return loss

#     def forward(self, sentence, intent, mask):
#         emissions = self.__get_lstm_features(sentence, intent)
#         intent_mask = torch.where(mask == 0, torch.tensor([INF]), torch.tensor([0.0]))
#         emissions = emissions + intent_mask
#         constraints = self.__create_constraints(mask, intent)
#         tag_sequence = self.crf.decode(emissions, constraints)
#         return tag_sequence
    

# # class ConstrainedCRF(CRF):
# #     def __init__(self, num_tags):
# #         super(ConstrainedCRF, self).__init__(num_tags)

# #     def decode(self, emissions: torch.Tensor, constraints: List[Tuple[torch.IntTensor, torch.IntTensor]], mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
# #         """Find the most likely tag sequence using Viterbi algorithm.

# #         Args:
# #             emissions (`~torch.Tensor`): Emission score tensor of size
# #                 ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
# #                 ``(batch_size, seq_length, num_tags)`` otherwise.
# #             mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
# #                 if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

# #         Returns:
# #             List of list containing the best tag sequence for each batch.
# #         """
# #         self._validate(emissions, mask=mask)
# #         if mask is None:
# #             mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

# #         if self.batch_first:
# #             emissions = emissions.transpose(0, 1)
# #             mask = mask.transpose(0, 1)

# #         return self._viterbi_decode(emissions, mask, constraints)

# #     def _viterbi_decode(self, emissions: torch.FloatTensor, mask: torch.ByteTensor, constraints: List[Tuple[torch.IntTensor, torch.IntTensor]]) -> List[List[int]]:
# #         print("Viterbi Decode")
# #         '''
# #                 override the viterbi decode function to include the constraints
# #             '''
# #         assert emissions.dim() == 3 and mask.dim(
# #         ) == 2  # (sequence_length, batch_size, num_tags)
# #         # (sequence_length, batch_size)
# #         assert emissions.shape[:2] == mask.shape
# #         assert emissions.size(2) == self.num_tags
# #         assert mask[0].all()

# #         sequence_length, batch_size = mask.shape

# #         constrained_transitions = self.transitions.clone()
# #         constrainted_satrt_transitions = self.start_transitions.clone()
# #         for constraint in constraints:
# #             if constraint[0] == -1:
# #                 constrainted_satrt_transitions[constraint[1]] = INF
# #             else:
# #                 constrained_transitions[constraint[0], constraint[1]] = INF

# #         # start transition and first emission
# #         # tensor of size (batch_size, num_tags)
# #         # constraints for start transition
# #         # score = self.start_transitions + emissions[0]
# #         score = constrainted_satrt_transitions + emissions[0]

# #         backpointers = []

# #         for i in range(1, sequence_length):
# #             # Broadcast viterbi score for every possible next tag
# #             # shape: (batch_size, num_tags, 1)
# #             broadcast_score = score.unsqueeze(2)

# #             # Broadcast emission score for every possible current tag
# #             # shape: (batch_size, 1, num_tags)
# #             broadcast_emission = emissions[i].unsqueeze(1)

# #             # Compute the score tensor of size (batch_size, num_tags, num_tags) where
# #             # for each sample, entry at row i and column j stores the score of the best
# #             # tag sequence so far that ends with transitioning from tag i to tag j and emitting
# #             # shape: (batch_size, num_tags, num_tags)
# #             # next_score = broadcast_score + self.transitions + broadcast_emission
# #             next_score = broadcast_score + constrained_transitions + broadcast_emission

# #             # print("Next Score Shape:", next_score[0].shape)

# #             # Apply the constraints
# #             # for constraint in constraints:
# #             #     # print("Constraint:", constraint)
# #             #     next_score[0][constraint[0], constraint[1]] = INF

# #             # Find the maximum score over all possible current tag
# #             # shape: (batch_size, num_tags)
# #             next_score, indices = next_score.max(dim=1)

# #             # Set score to the next score if this timestep is valid (mask == 1)
# #             # and save the index that produces the next score
# #             # shape: (batch_size, num_tags)
# #             score = torch.where(mask[i].unsqueeze(1), next_score, score)
# #             backpointers.append(indices)

# #         # End transition score
# #         # shape: (batch_size, num_tags)
# #         score += self.end_transitions

# #         # Now, compute the best path for each sample

# #         # shape: (batch_size,)
# #         seq_ends = mask.long().sum(dim=0) - 1
# #         best_tags_list = []

# #         for idx in range(batch_size):
# #             # Find the tag which maximizes the score at the last timestep; this is our best tag
# #             # for the last timestep
# #             _, best_last_tag = score[idx].max(dim=0)
# #             best_tags = [best_last_tag.item()]

# #             # We trace back where the best last tag comes from, append that to our best tag
# #             # sequence, and trace it back again, and so on
# #             for hist in reversed(backpointers[:seq_ends[idx]]):
# #                 best_last_tag = hist[idx][best_tags[-1]]
# #                 best_tags.append(best_last_tag.item())

# #             # Reverse the order because we start from the last timestep
# #             best_tags.reverse()
# #             best_tags_list.append(best_tags)

# #         return best_tags_list

class ConstrainedCRF(CRF):
    def __init__(self, num_tags):
        super(ConstrainedCRF, self).__init__(num_tags)

    def decode(self, emissions: torch.Tensor, constraints: List[Tuple[torch.IntTensor, torch.IntTensor]], mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

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
            # print("Viterbi Decode")
            '''
                override the viterbi decode function to include the constraints
            '''
            assert emissions.dim() == 3 and mask.dim() == 2 # (sequence_length, batch_size, num_tags)
            assert emissions.shape[:2] == mask.shape # (sequence_length, batch_size)
            assert emissions.size(2) == self.num_tags
            assert mask[0].all()

            sequence_length, batch_size = mask.shape

            constrained_transitions = self.transitions.clone()
            # Apply the constraints
            for constraint in constraints:
                # print("Constraint:", constraint)
                constrained_transitions[constraint[0], constraint[1]] = INF


            # start transition and first emission
            # tensor of size (batch_size, num_tags)
            score = self.start_transitions + emissions[0]

            backpointers = []

            for i in range(1, sequence_length):
                # Broadcast viterbi score for every possible next tag
                # shape: (batch_size, num_tags, 1)
                broadcast_score = score.unsqueeze(2)


                # Broadcast emission score for every possible current tag
                # shape: (batch_size, 1, num_tags)
                broadcast_emission = emissions[i].unsqueeze(1)

                # Compute the score tensor of size (batch_size, num_tags, num_tags) where
                # for each sample, entry at row i and column j stores the score of the best
                # tag sequence so far that ends with transitioning from tag i to tag j and emitting
                # shape: (batch_size, num_tags, num_tags)
                next_score = broadcast_score + constrained_transitions + broadcast_emission

                # print("Next Score Shape:", next_score[0].shape)


                # Find the maximum score over all possible current tag
                # shape: (batch_size, num_tags)
                next_score, indices = next_score.max(dim=1)

                # Set score to the next score if this timestep is valid (mask == 1)
                # and save the index that produces the next score
                # shape: (batch_size, num_tags)
                score = torch.where(mask[i].unsqueeze(1), next_score, score)
                backpointers.append(indices)


            # End transition score
            # shape: (batch_size, num_tags)
            score += self.end_transitions


            # Now, compute the best path for each sample

            # shape: (batch_size,)
            seq_ends = mask.long().sum(dim=0) - 1
            best_tags_list = []

            for idx in range(batch_size):
                # Find the tag which maximizes the score at the last timestep; this is our best tag
                # for the last timestep
                _, best_last_tag = score[idx].max(dim=0)
                best_tags = [best_last_tag.item()]

                # We trace back where the best last tag comes from, append that to our best tag
                # sequence, and trace it back again, and so on
                for hist in reversed(backpointers[:seq_ends[idx]]):
                    best_last_tag = hist[idx][best_tags[-1]]
                    best_tags.append(best_last_tag.item())

                # Reverse the order because we start from the last timestep
                best_tags.reverse()
                best_tags_list.append(best_tags)

            return best_tags_list


class NERModel(nn.Module):
    def __init__(self, vocab_size, word_embedding_dim=100, intent_embedding_dim=100, hidden_dim=64, output_dim=45, number_of_intents=23, index_to_tag=None):

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
        self.intent_embedding = nn.Embedding(number_of_intents, intent_embedding_dim)
        self.lstm = nn.LSTM(word_embedding_dim + intent_embedding_dim, hidden_dim // 2, bidirectional=True, dropout=0.2)
        self.hidden_to_tag = nn.Linear(hidden_dim, output_dim)
        self.crf = ConstrainedCRF(output_dim)


    def __create_constraints(self, mask):
        constraints = []
        one_indices = torch.where(mask == 1)[0]
        zero_indices = torch.where(mask == 0)[0]
        # print(one_indices)
        # print(zero_indices)
        for i in one_indices:
            for j in zero_indices:
                constraints.append((i, j))
                constraints.append((j, i))

            if self.index_to_tag[i.item()] == 'O':
                for j in one_indices:
                    if self.index_to_tag[j.item()][0] == 'I':
                        constraints.append((i, j))

        for i in zero_indices:
            for j in zero_indices:
                constraints.append((i, j))

        return constraints

    def __init_hidden(self):
        # initialize hidden state
        return (torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2))

    def __get_lstm_features(self, sentence, intent):
        self.hidden = self.__init_hidden()
        word_embeddings = self.word_embedding(
            sentence).view(len(sentence), 1, -1)
        intent_embeddings = self.intent_embedding(
            intent).view(1, 1, -1).repeat(len(sentence), 1, 1)
        embeddings = torch.cat((word_embeddings, intent_embeddings), dim=2)
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        lstm_features = self.hidden_to_tag(lstm_out)
        return lstm_features

    def neg_log_likelihood(self, sentence, tags, intent, mask):
        # (sentence_length, batch_size, number of tags) # emissions where each word in the sequence coressponds to probability for tags
        emissions = self.__get_lstm_features(sentence, intent)
        intent_mask = torch.where(
            mask == 0, torch.tensor([INF]), torch.tensor([0.0]))
        emissions = emissions + intent_mask
        tags = tags.view(-1, 1)
        loss = -self.crf(emissions, tags)
        return loss

    def forward(self, sentence, intent, mask):
        emissions = self.__get_lstm_features(sentence, intent)
        intent_mask = torch.where(mask == 0, torch.tensor([INF]), torch.tensor([0.0]))
        emissions = emissions + intent_mask
        constraints = self.__create_constraints(mask)
        tag_sequence = self.crf.decode(emissions, constraints)
        return tag_sequence


def train():
    pass


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
