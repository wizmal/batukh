from batukh.torch.segmenter import BaseProcessor
import pytest
import os


@pytest.mark.parametrize("path, actual_value", [
    # simple case
    (['1-2020-9-11-21-4-6.pt',
      '2-2020-9-11-2-3-7.pth',
      '3-2020-9-11-4-34-12.pt',
      '4-2020-9-12-4-54-52.pth'], '4-2020-9-12-4-54-52.pth'),

    # lower epoch higher time
    (['2-2019-8-21-5-8-33.pt',
      '1-2019-8-21-5-8-34.pt',
      '3-2019-8-21-3-22-22.pth',
      '4-2019-8-20-12-34-55.pt'], '1-2019-8-21-5-8-34.pt',),

    # equal epochs different time
    (['2-2019-12-30-22-33-51.pt',
      '2-2019-12-30-22-15-22.pt',
      '1-2019-12-29-12-34-54.pth',
      '3-2019-12-29-22-34-52.pt'], '2-2019-12-30-22-33-51.pt'),

    # equal time different epochs
    (['3-2020-05-21-12-45-22.pt',
      '2-2020-05-21-12-45-22.pt',
      '1-2020-05-20-21-44-23.pth'], '3-2020-05-21-12-45-22.pt')
])
def test_get_latest_ckpt_path(monkeypatch, path, actual_value):

    # supply fake os.listdir
    def mock_listdir(path):
        return path

    monkeypatch.setattr(os, "listdir", mock_listdir)

    ouput = BaseProcessor.get_latest_ckpt_path(None, path)
    assert ouput == actual_value
