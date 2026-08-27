"""Pure-Python inventory of rosbag2 observation-source metadata.

This module reads rosbag2 metadata only. It does not deserialize messages, infer
extrinsics, choose a preferred source, or export rays.
"""

from __future__ import annotations


def _classify_topic(name, type_name):
    name_lower = str(name).lower()
    type_name = str(type_name)
    roles = []

    if type_name == "sensor_msgs/msg/PointCloud2":
        family = "pointcloud2"
        roles.append("lidar_returns")
    elif type_name.endswith("/CustomMsg") and (
        "livox" in type_name.lower() or "livox" in name_lower
    ):
        family = "livox_custom"
        roles.append("lidar_returns")
    elif type_name == "sensor_msgs/msg/Imu":
        family = "imu"
        roles.append("imu")
    elif type_name in {
        "nav_msgs/msg/Odometry",
        "geometry_msgs/msg/PoseStamped",
        "geometry_msgs/msg/PoseWithCovarianceStamped",
    }:
        family = "pose"
        roles.append("pose_or_odometry")
    elif type_name == "tf2_msgs/msg/TFMessage":
        family = "tf"
        roles.append("transform")
    else:
        family = "other"
        # Keep name-based hints deliberately narrow and diagnostic only.
        if any(token in name_lower for token in ("lidar", "pointcloud", "points", "cloud")):
            roles.append("lidar_returns")
        if any(token in name_lower for token in ("odom", "pose", "trajectory")):
            roles.append("pose_or_odometry")

    return family, roles


def inventory_rosbag2_metadata(payload):
    """Return a conservative source inventory from parsed rosbag2 metadata YAML."""
    if not isinstance(payload, dict):
        raise TypeError("metadata payload must be a mapping")
    info = payload.get("rosbag2_bagfile_information")
    if not isinstance(info, dict):
        raise ValueError("missing rosbag2_bagfile_information mapping")

    topic_records = info.get("topics_with_message_count", [])
    if not isinstance(topic_records, list):
        raise ValueError("topics_with_message_count must be a list")

    topics = []
    candidates = {
        "lidar_returns": [],
        "pose_or_odometry": [],
        "transform": [],
        "imu": [],
    }
    for record in topic_records:
        if not isinstance(record, dict):
            continue
        metadata = record.get("topic_metadata", {})
        if not isinstance(metadata, dict):
            continue
        name = str(metadata.get("name", ""))
        type_name = str(metadata.get("type", ""))
        family, roles = _classify_topic(name, type_name)
        item = {
            "name": name,
            "type": type_name,
            "serialization_format": str(metadata.get("serialization_format", "")),
            "message_count": int(record.get("message_count", 0)),
            "message_family": family,
            "candidate_roles": roles,
        }
        topics.append(item)
        for role in roles:
            if role in candidates and name not in candidates[role]:
                candidates[role].append(name)

    return {
        "schema_version": 1,
        "storage_identifier": str(info.get("storage_identifier", "")),
        "metadata_version": int(info.get("version", 0)),
        "message_count": int(info.get("message_count", 0)),
        "topics": topics,
        "candidates": candidates,
        "automatic_source_selection": False,
    }
