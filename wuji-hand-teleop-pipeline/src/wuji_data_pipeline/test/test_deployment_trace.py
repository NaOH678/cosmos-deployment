import json

from wuji_data_pipeline.deployment_trace import DeploymentTraceWriter


def test_deployment_trace_writer_persists_ordered_jsonl(tmp_path):
    writer = DeploymentTraceWriter(
        tmp_path,
        session_id="0123456789abcdef",
        queue_size=8,
        flush_interval_s=0.01,
    )
    writer.record("prefetch_request", request_id=4, remaining_actions=5)
    writer.record("pending_activate", request_id=4, skipped_actions=5)
    path = writer.path
    writer.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "prefetch_request",
        "pending_activate",
        "trace_summary",
    ]
    assert records[0]["request_id"] == 4
    assert records[1]["skipped_actions"] == 5
    assert writer.dropped_events == 0

    assert records[-1]["complete"] is True
    assert writer.closed_cleanly is True
    assert writer.writer_error == ""
