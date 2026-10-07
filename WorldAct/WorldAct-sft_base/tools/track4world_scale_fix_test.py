import torch
from track4world_scale_fix import ChunkScaleLedger, correct_cached_depth


def test_inverse_and_boundary_units():
    first = torch.full((2, 3, 2, 2), 0.2)
    second = torch.full((3, 3, 2, 2), 0.8)
    ledger = ChunkScaleLedger()
    normalized = torch.cat([ledger.normalize(first), ledger.normalize(second)])
    restored = ledger.restore(normalized)
    expected = torch.cat([first, second])
    torch.testing.assert_close(restored, expected)
    torch.testing.assert_close(restored[2] - restored[1], torch.full((3, 2, 2), 0.6))
    legacy = normalized * (ledger.divisors[-1] - 1e-6)
    corrected = correct_cached_depth(legacy[:2], ledger.divisors[0].item() - 1e-6, ledger.divisors[-1].item() - 1e-6)
    torch.testing.assert_close(corrected, first)
    assert not torch.allclose(legacy[:2], first)


def test_world_and_camera_pose_inverse():
    ledger = ChunkScaleLedger()
    chunks = []
    expected = []
    for value in (.1, 1.2):
        camera = torch.full((2, 3, 2, 2), value)
        world = camera + .3
        poses = torch.eye(4).repeat(2, 1, 1)
        poses[:, :3, 3] = value
        expected.append((camera, world, poses))
        chunks.append(ledger.normalize_geometry(camera, world, poses))
    actual = ledger.restore_geometry(*(torch.cat([chunk[i] for chunk in chunks]) for i in range(3)))
    for i, value in enumerate(actual):
        torch.testing.assert_close(value, torch.cat([chunk[i] for chunk in expected]))
