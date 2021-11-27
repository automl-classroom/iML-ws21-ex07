import sys
import os  # noqa
sys.path.insert(0, ".")  # noqa

import torch
torch.manual_seed(0)

from classifiers.distilbert import Selector, Predictor
from tests.config import WORKING_DIR
from utils.dataset import SentimentRationaleDataset, _custom_collate, download_and_preprocess_dataset


module = __import__(f"{WORKING_DIR}.select_and_predict", fromlist=[
    'train_selector_model', 'select', 'predict'])

download_and_preprocess_dataset()
ds_train = SentimentRationaleDataset('train', limit=100)
ds_val = SentimentRationaleDataset('dev', limit=100)

dl_train = ds_train.get_dataloader()
dl_val = ds_val.get_dataloader()

selector_model = Selector()
predictor_model = Predictor()


def test_train_selector_model():
    train_loss, train_acc, val_loss, val_acc = module.train_selector_model(selector_model, dl_train, dl_val)
    assert torch.tensor(train_loss).isclose(torch.tensor(0.16), atol=2e-2)
    assert torch.tensor(train_acc).isclose(torch.tensor(0.93), atol=2e-2)
    assert torch.tensor(val_loss).isclose(torch.tensor(0.10), atol=2e-2)
    assert torch.tensor(val_acc).isclose(torch.tensor(0.96), atol=2e-2)


def test_select():
    selection = module.select(selector_model, dl_val)
    assert isinstance(selection, list)
    assert isinstance(selection[0], dict)
    assert sorted(list(selection[0].keys())) == ['label', 'text']
    assert selection[0]['label'] == 1
    assert selection[1]['label'] == 0
    assert selection[0]['text'][0] == '[MASK]'
    assert selection[0]['text'][1] == '[MASK]'
    assert selection[0]['text'][6] == 'me'
    assert selection[7]['text'][2] == 'fooled'


def test_predict():
    ds_val_masked = [
        {'text': ['[MASK]', 'gorgeous', 'high', '-', 'spirited', 'musical', '[MASK]', '[MASK]', '[MASK]', 'exquisitely', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', 'high', 'drama', '[MASK]'],
        'label': 1},
        {'text': ['[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', 'me', 'miss', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', 'optimistic', '[MASK]', '[MASK]', '[MASK]', '[MASK]', 'hope', '[MASK]', 'popular', '[MASK]', '[MASK]', '[MASK]'],
        'label': 1}
    ]

    dl = torch.utils.data.DataLoader(ds_val_masked, batch_size=1, collate_fn=_custom_collate)
    predictions = module.predict(predictor_model, dl)
    assert isinstance(predictions, list)
    assert len(predictions) == 2
    assert predictions == [1, 1]

    dl = torch.utils.data.DataLoader(ds_val_masked, batch_size=2, collate_fn=_custom_collate)
    predictions = module.predict(predictor_model, dl)
    assert isinstance(predictions, list)
    assert len(predictions) == 2
    assert predictions == [1, 1]


if __name__ == "__main__":
    test_train_selector_model()
    test_select()
    test_predict()
