from pathlib import Path

from wuji_data_pipeline.hand_teleop_session import _required_enable_nodes


def test_hand_only_enable_nodes_follow_selected_side():
    assert _required_enable_nodes("right") == {
        "/right_hand/wujihand_driver",
        "/wujihand_controller_right",
    }


def test_hand_only_script_uses_supervised_session_not_auto_enable():
    root = Path(__file__).parents[2]
    source = (root / "scripts" / "start_teleop_hand_only.sh").read_text()

    assert "hand_teleop_session" in source
    assert "auto_enable:=true" not in source
