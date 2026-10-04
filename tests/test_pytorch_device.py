import pytest

torch = pytest.importorskip("torch")

from frameworks import saps_pytorch  # noqa: E402
from frameworks.saps_pytorch import PytorchFramework  # noqa: E402


@pytest.mark.parametrize(
    "saps_device,cuda,expected",
    [(None, True, "cpu"), ("cpu", True, "cpu"), ("gpu", True, "cuda")],
)
def test_saps_device_sets_torch_default_device(
    monkeypatch, saps_device, cuda, expected
):
    if saps_device is None:
        monkeypatch.delenv("SAPS_DEVICE", raising=False)
    else:
        monkeypatch.setenv("SAPS_DEVICE", saps_device)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    devices = []
    monkeypatch.setattr(saps_pytorch.torch, "set_default_device", devices.append)

    assert PytorchFramework().device == expected
    assert devices == [expected]


def test_gpu_device_without_cuda_fails_loudly(monkeypatch):
    monkeypatch.setenv("SAPS_DEVICE", "gpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(saps_pytorch.torch, "set_default_device", lambda device: None)

    with pytest.raises(RuntimeError, match="CUDA"):
        PytorchFramework()
