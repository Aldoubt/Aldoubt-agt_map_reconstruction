import json
from pathlib import Path
import subprocess
import sys

import yaml


def test_rosbag_observation_inventory_cli_reads_metadata_directory(tmp_path):
    bag = tmp_path / "bag"
    bag.mkdir()
    metadata = {
        "rosbag2_bagfile_information": {
            "version": 5,
            "storage_identifier": "sqlite3",
            "message_count": 20,
            "topics_with_message_count": [
                {
                    "topic_metadata": {
                        "name": "/livox/lidar",
                        "type": "livox_ros_driver2/msg/CustomMsg",
                        "serialization_format": "cdr",
                    },
                    "message_count": 10,
                },
                {
                    "topic_metadata": {
                        "name": "/Odometry",
                        "type": "nav_msgs/msg/Odometry",
                        "serialization_format": "cdr",
                    },
                    "message_count": 10,
                },
            ],
        }
    }
    (bag / "metadata.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    output = tmp_path / "inventory.json"
    script = Path(__file__).resolve().parents[1] / "tools" / "inventory_rosbag_observation_sources.py"

    completed = subprocess.run(
        [sys.executable, str(script), str(bag), "--output", str(output)],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["candidates"]["lidar_returns"] == ["/livox/lidar"]
    assert result["candidates"]["pose_or_odometry"] == ["/Odometry"]
    assert result["automatic_source_selection"] is False
    assert "automatic_source_selection: false" in completed.stdout
