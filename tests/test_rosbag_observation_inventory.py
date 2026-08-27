from agt_map_reconstruction.io.rosbag_observation_inventory import (
    inventory_rosbag2_metadata,
)


def _metadata_payload():
    return {
        "rosbag2_bagfile_information": {
            "version": 5,
            "storage_identifier": "sqlite3",
            "message_count": 1234,
            "topics_with_message_count": [
                {
                    "topic_metadata": {
                        "name": "/livox/lidar",
                        "type": "livox_ros_driver2/msg/CustomMsg",
                        "serialization_format": "cdr",
                    },
                    "message_count": 100,
                },
                {
                    "topic_metadata": {
                        "name": "/cloud_registered",
                        "type": "sensor_msgs/msg/PointCloud2",
                        "serialization_format": "cdr",
                    },
                    "message_count": 100,
                },
                {
                    "topic_metadata": {
                        "name": "/Odometry",
                        "type": "nav_msgs/msg/Odometry",
                        "serialization_format": "cdr",
                    },
                    "message_count": 100,
                },
                {
                    "topic_metadata": {
                        "name": "/tf",
                        "type": "tf2_msgs/msg/TFMessage",
                        "serialization_format": "cdr",
                    },
                    "message_count": 900,
                },
                {
                    "topic_metadata": {
                        "name": "/livox/imu",
                        "type": "sensor_msgs/msg/Imu",
                        "serialization_format": "cdr",
                    },
                    "message_count": 34,
                },
            ],
        }
    }


def test_inventory_classifies_observation_sources_without_selecting_one():
    result = inventory_rosbag2_metadata(_metadata_payload())

    assert result["storage_identifier"] == "sqlite3"
    assert result["message_count"] == 1234
    assert result["automatic_source_selection"] is False

    by_name = {item["name"]: item for item in result["topics"]}
    assert by_name["/livox/lidar"]["candidate_roles"] == ["lidar_returns"]
    assert by_name["/livox/lidar"]["message_family"] == "livox_custom"
    assert by_name["/cloud_registered"]["candidate_roles"] == ["lidar_returns"]
    assert by_name["/cloud_registered"]["message_family"] == "pointcloud2"
    assert by_name["/Odometry"]["candidate_roles"] == ["pose_or_odometry"]
    assert by_name["/tf"]["candidate_roles"] == ["transform"]
    assert by_name["/livox/imu"]["candidate_roles"] == ["imu"]

    assert result["candidates"]["lidar_returns"] == [
        "/livox/lidar",
        "/cloud_registered",
    ]
    assert result["candidates"]["pose_or_odometry"] == ["/Odometry"]
    assert result["candidates"]["transform"] == ["/tf"]


def test_inventory_keeps_unknown_message_types_visible():
    payload = _metadata_payload()
    payload["rosbag2_bagfile_information"]["topics_with_message_count"].append(
        {
            "topic_metadata": {
                "name": "/vendor/something",
                "type": "vendor_msgs/msg/Foo",
                "serialization_format": "cdr",
            },
            "message_count": 5,
        }
    )

    result = inventory_rosbag2_metadata(payload)
    item = next(topic for topic in result["topics"] if topic["name"] == "/vendor/something")
    assert item["candidate_roles"] == []
    assert item["message_family"] == "other"
