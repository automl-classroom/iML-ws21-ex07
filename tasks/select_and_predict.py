import sys
import os  # noqa
sys.path.insert(0, ".")  # noqa

import torch
torch.manual_seed(0)

from utils.dataset import (
        SentimentRationaleDataset,
        tokenize,
        decode,
        pad_token_id,
        mask_token_id,
        cls_token_id,
        sep_token_id,
        _custom_collate
    )
from classifiers.distilbert import Selector, Predictor
from tqdm.auto import tqdm


"""
Some documentation for imports:


tokenize (function):
    Use this function to tokenize the input text, and optionally align the corresponding rationales.

    Parameters:
        text (List[List[str]]):
            A batch of text as returned by the dataloaders.
        Optional: rationale (List[List[int]]):
            A batch of rationale masks as returned by the dataloaders.
            Required when using the rationales as labels, as they have to remain aligned with the text tokens.
    Returns:
        tokenized_inputs (dict):
            A dict containing the tokenized text (key='input_ids'), an attention_mask (key='attention_mask') and aligned rationales (key='rationales) if passed.
            Rationale labels that belong to tokens that not belong to the text are labeld with -100.


decode (function):
    Use this function to turn tokenized input_ids back to text.

    Parameters:
        input_ids (torch.tensor): A batch of input ids.
    Returns:
        text (str): decoded text.


{pad, mask, cls, sep}_token_id (int):
    The token_id representing the [PAD], [MASK], [CLS], [SEP] token, respectively.

"""



def train_selector_model(selector_model, dl_train, dl_val):
    """
    Trains the given selector model for one epoch, then validates the model.
    Essentially, the goal of the selector model is to predict a mask such that only the important tokens are revealed.
    For example, for the positive movie review
        `A georgeous movie .`
    the prediction could be `0, 1, 0, 0`.
    For each token in the input sequence, the model predicts 0 if the token should be masked and 1 if the token should be revealed.
    In this exercise, we train the selector in a supervised manner, using annotated rationale data (in the form of binary masks).

    The dataloaders for the selector return batches in the form of dicts, with the following structure:
        'text': List[List[str]]:
            A batch of movie review text.
            Each review is a List of tokens.
        'rationale': List[List[int]]:
            A batch of rationale masks.
            Each rationale is a List representing a binary mask over tokens (length = num of tokens in text).
        'label': List[int]: 
            A batch of labels, either 0 (negative) or 1(positive).
            Not relevant for training the selector, as here the rationale masks are used as groundtruth.

    Parameters:
        selector_model (Selector): 
            A token classification model based on DistilBERT.
            For each token in the input sequence, the model predicts whether it should be masked (0) or not (1).
            The selector_model is also a torch.nn.Module, so you can call its forward method as
                selector_model(input_ids, attention_mask)
            Both of these inputs can be created by applying the `tokenize` function on a batch returned by the dataloaders.

        dl_train (torch.utils.data.DataLoader): The dataloader containing the training instances in batches.
        dl_val (torch.utils.data.DataLoader): The dataloader containing the validation instances in batches.
    
    Returns:
        epoch_train_loss (float): The average loss over the batches seen during training.
        epoch_train_acc (float): The average accuracy over the batches seen during training.
        epoch_val_loss (float): The average loss over the batches seen during validation.
        epoch_val_acc (float): The average accuracy over the batches seen during validation.
    """
    optimizer = torch.optim.AdamW(selector_model.parameters(), lr=1e-5)

    def train_one_epoch():
        """
        Trains the selector model for one epoch. Uses torch.nn.functional.cross_entropy loss.
        The training can be summarized as:
            for each batch:
                - tokenize the batch and rationales
                - set gradients to zero
                - forward pass
                - computing the loss.
                - backward pass & optimization step
                - obtaining the predicted labels (binary)
                - computing the accuracy.

        Returns:
            train_losses (List[float]): A list containing the loss of each batch seen during training.
            train_accs (List[float]) A list containing the accuracy of each batch seen during training.

        Hints: 
            - Use `.item()` before appending the loss / accuracy of a batch to the correpsonding list.
            - torch.nn.functional.cross_entropy already automatically ignores inputs labeled with -100.
            - When computing accuracy, only include the input_ids belonging to the text.
                These are all the tokens for which the tokenized rationale mask is != -100.
            
        """
        selector_model.train()

        # TODO

        return None

    def validate():
        """
        Validates the selector model. Uses torch.nn.functional.cross_entropy loss.
        The validation can be summarized as:
            for each batch:
                - tokenize the batch and rationales
                - forward pass
                - computing the loss.
                - obtaining the predicted labels (binary)
                - computing the accuracy.

        Returns:
            val_losses (List[float]): A list containing the loss of each batch seen during validation.
            val_accs (List[float]) A list containing the accuracy of each batch seen during validation.

        Hint: See train_one_epoch(). Optionally use `with torch.no_grad():` to disable gradient tracking (not needed during eval).
            
        """
        selector_model.eval()

        # TODO

        return None


    #TODO
    
    return None


def select(selector_model, dl):
    """
    Uses the selector_model to predict the masked text for all instances in the dataloader.
    The selection can be summarized as:
        for each batch:
            - tokenize the text and the rationale
            - forward pass
            - obtaining the predicted labels (binary)
            for each instance in the batch:
                - selecting the relevant input_ids (all input_ids that are not cls_token_id, sep_token_id, pad_token_id])
                - applying the predicted mask to the relevant input_ids, by replacing all input_ids for which the mask=0 with mask_token_id
                - decoding the input_ids back to (now masked) text using the `decode` function

    Parameters:
        selector_model (Selector): 
            The selector model used for prediction.
        dl (torch.utils.data.DataLoader):
            The dataloader containing the instances to predict.

    Returns:
        selections (List[dict]):
            A list containing one dict for every instance in the dataloader.
            Each dict has two keys:
                'text': A list of tokens (as in the dataloader). Some tokens are replaced with the mask token [MASK]
                'label': The label of the instance (as in the dataloader).

    """
    return None


def predict(predictor_model, dl):
    """
    Predicts the sentiment label for the instances in the dataloader.

    Parameters:
        predictor_model (Predictor):
            A sequence classification model based on DistilBERT.
            For each input sequence, the model predicts whether it is negative (0) or positive (1).
            The predictor model is also a torch.nn.Module, so you can call its forward method as
                predictor_model(input_ids, attention_mask)
            Both of these inputs can be created by applying the `tokenize` function on a batch returned by the dataloaders.

        dl (torch.utils.data.DataLoader):
            The dataloader containing the instances to predict.
            The instances in the dataloader are the results of the `select` function.

    Returns:
        predictions (List[int]): A list containing the predicted labels (0 or 1) for each instance in the dataloader.
    """
    return None


if __name__ == "__main__":

    ds_train = SentimentRationaleDataset('train', limit=1000)
    ds_val = SentimentRationaleDataset('dev', limit=100)

    dl_train = ds_train.get_dataloader()
    dl_val = ds_val.get_dataloader()

    selector_model = Selector()

    print('Running `train_selector_model` ...')
    train_loss, train_acc, val_loss, val_acc = train_selector_model(selector_model, dl_train, dl_val)


    print('Running `select` ...')
    ds_val_masked = select(selector_model, dl_val)

    predictor_model = Predictor()
    dl = torch.utils.data.DataLoader(ds_val_masked, batch_size=8, collate_fn=_custom_collate)

    print('Running `predict` ...')
    predictions = predict(predictor_model, dl)

    print('=' * 80)
    print('Examples:')
    for i in range(0, 4):
        print('-' * 80)
        print(f'Instance: {" ".join(ds_val[i]["text"])}')
        print(f'Selection: {" ".join(ds_val_masked[i]["text"])}')
        print(f'Prediction: {"positive" if predictions[i] else "negative"}')
        print(f'Groundtruth: {"positive" if ds_val[i]["label"] else "negative"}')
        print('-' * 80)
