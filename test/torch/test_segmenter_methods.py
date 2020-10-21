from batukh.torch.segmenter import BaseProcessor
import pytest
import os


@pytest.mark.parametrize("path, actual_value", [
    (['1-2020-9-11-21-4-6.pt',
      '2-2020-9-11-2-3-7.pth',
      '3-2020-9-11-4-34-12.pt',
      '4-2020-9-12-4-54-52.pth'], '4-2020-9-12-4-54-52.pth')
])
def test_get_latest_ckpt_path(path, actual_value):

    # supply fake os module
    class os:

        @staticmethod
        def listdir(path):
            return path

    ouput = BaseProcessor.get_latest_ckpt_path(None, path)
    assert ouput == actual_value
