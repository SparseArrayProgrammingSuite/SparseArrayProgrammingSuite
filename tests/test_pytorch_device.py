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


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device available"
)


@pytest.fixture
def cpu_xp(monkeypatch):
    # Keep the process default device; the tests place tensors explicitly.
    monkeypatch.delenv("SAPS_DEVICE", raising=False)
    monkeypatch.setattr(saps_pytorch.torch, "set_default_device", lambda device: None)
    return PytorchFramework()


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32, torch.int64])
def test_integer_matmul_on_cuda_matches_cpu(cpu_xp, dtype):
    # CUDA has no integer GEMM; uint8 products also wrap, as on the CPU.
    generator = torch.Generator().manual_seed(0)
    x1 = torch.randint(0, 200, (5, 7), dtype=dtype, generator=generator)
    x2 = torch.randint(0, 200, (7, 3), dtype=dtype, generator=generator)

    result = cpu_xp.matmul(x1.cuda(), x2.cuda())

    assert result.device.type == "cuda"
    assert result.dtype == dtype
    torch.testing.assert_close(result.cpu(), x1 @ x2)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bool, torch.int64])
def test_integer_einsum_on_cuda_matches_cpu(cpu_xp, dtype):
    generator = torch.Generator().manual_seed(0)
    a = torch.randint(0, 3, (4, 6), generator=generator).to(dtype)
    b = torch.randint(0, 3, (6, 5), generator=generator).to(dtype)
    program = "C[i,j] += A[i,k] * B[k,j]"

    result = cpu_xp.einsum(program, A=a.cuda(), B=b.cuda())

    assert result.device.type == "cuda"
    torch.testing.assert_close(result.cpu(), cpu_xp.einsum(program, A=a, B=b))
