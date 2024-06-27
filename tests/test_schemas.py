"""
Unit tests for Pydantic schemas in src/models/schemas.py.

Tests cover validation, cross-field validators, and serialization.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    AcknowledgeRequest,
    AlertSchema,
    BatchDetectRequest,
    BatchDetectResponse,
    BoundingBoxSchema,
    ComponentHealth,
    DetectRequest,
    DetectResponse,
    DetectionSchema,
    FWIComponentsSchema,
    HealthResponse,
    IncidentSchema,
    IncidentUpdateSchema,
    RiskZoneRequest,
    RiskZoneResponse,
    SpreadHorizonSchema,
    StreamFrame,
    StreamSubscribeMessage,
    TransitionRequest,
)
import time


# ---------------------------------------------------------------------------
# BoundingBoxSchema
# ---------------------------------------------------------------------------


class TestBoundingBoxSchema:
    def test_valid_bbox(self):
        b = BoundingBoxSchema(x1=0.1, y1=0.2, x2=0.5, y2=0.8)
        assert b.x1 == 0.1

    def test_x2_must_be_greater_than_x1(self):
        with pytest.raises(ValidationError, match="x2 must be greater than x1"):
            BoundingBoxSchema(x1=0.5, y1=0.2, x2=0.3, y2=0.8)

    def test_y2_must_be_greater_than_y1(self):
        with pytest.raises(ValidationError, match="y2 must be greater than y1"):
            BoundingBoxSchema(x1=0.1, y1=0.8, x2=0.5, y2=0.2)

    def test_coordinates_must_be_normalized(self):
        with pytest.raises(ValidationError):
            BoundingBoxSchema(x1=-0.1, y1=0.0, x2=0.5, y2=0.5)

    def test_coordinates_max_one(self):
        with pytest.raises(ValidationError):
            BoundingBoxSchema(x1=0.0, y1=0.0, x2=1.5, y2=1.0)

    def test_exactly_equal_x1_x2_raises(self):
        with pytest.raises(ValidationError):
            BoundingBoxSchema(x1=0.5, y1=0.0, x2=0.5, y2=1.0)


# ---------------------------------------------------------------------------
# DetectRequest
# ---------------------------------------------------------------------------


class TestDetectRequest:
    def test_valid_detect_request(self):
        req = DetectRequest(image_b64="abc123")
        assert req.image_b64 == "abc123"
        assert req.run_spread_prediction is False

    def test_empty_b64_raises(self):
        with pytest.raises(ValidationError):
            DetectRequest(image_b64="")

    def test_with_camera_id_and_coords(self):
        req = DetectRequest(
            image_b64="data",
            camera_id="cam-001",
            latitude=37.77,
            longitude=-119.55,
        )
        assert req.camera_id == "cam-001"
        assert req.latitude == pytest.approx(37.77)

    def test_latitude_out_of_range(self):
        with pytest.raises(ValidationError):
            DetectRequest(image_b64="data", latitude=95.0)

    def test_longitude_out_of_range(self):
        with pytest.raises(ValidationError):
            DetectRequest(image_b64="data", longitude=200.0)

    def test_spread_prediction_flag(self):
        req = DetectRequest(image_b64="data", run_spread_prediction=True)
        assert req.run_spread_prediction is True

    def test_default_spread_horizons(self):
        req = DetectRequest(image_b64="data")
        assert req.spread_horizons_minutes == [30, 60, 120]


# ---------------------------------------------------------------------------
# BatchDetectRequest
# ---------------------------------------------------------------------------


class TestBatchDetectRequest:
    def test_valid_batch(self):
        req = BatchDetectRequest(images_b64=["img1", "img2", "img3"])
        assert len(req.images_b64) == 3

    def test_empty_batch_raises(self):
        with pytest.raises(ValidationError):
            BatchDetectRequest(images_b64=[])

    def test_exceeds_max_batch_raises(self):
        with pytest.raises(ValidationError):
            BatchDetectRequest(images_b64=["x"] * 17)

    def test_exactly_16_ok(self):
        req = BatchDetectRequest(images_b64=["x"] * 16)
        assert len(req.images_b64) == 16


# ---------------------------------------------------------------------------
# RiskZoneRequest
# ---------------------------------------------------------------------------


class TestRiskZoneRequest:
    def test_defaults(self):
        req = RiskZoneRequest()
        assert req.detection_confidence == 0.0
        assert req.days_since_rain == 0
        assert req.vegetation_type == "unknown"
        assert req.month == 7

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            RiskZoneRequest(detection_confidence=1.5)

    def test_negative_days_raises(self):
        with pytest.raises(ValidationError):
            RiskZoneRequest(days_since_rain=-1)

    def test_month_out_of_range(self):
        with pytest.raises(ValidationError):
            RiskZoneRequest(month=13)

    def test_month_zero_raises(self):
        with pytest.raises(ValidationError):
            RiskZoneRequest(month=0)


# ---------------------------------------------------------------------------
# AcknowledgeRequest
# ---------------------------------------------------------------------------


class TestAcknowledgeRequest:
    def test_valid(self):
        req = AcknowledgeRequest(acknowledged_by="operator-1")
        assert req.acknowledged_by == "operator-1"

    def test_empty_by_raises(self):
        with pytest.raises(ValidationError):
            AcknowledgeRequest(acknowledged_by="")

    def test_long_name_raises(self):
        with pytest.raises(ValidationError):
            AcknowledgeRequest(acknowledged_by="x" * 101)


# ---------------------------------------------------------------------------
# TransitionRequest
# ---------------------------------------------------------------------------


class TestTransitionRequest:
    def test_valid(self):
        req = TransitionRequest(new_status="RESPONDING", updated_by="dispatch")
        assert req.new_status == "RESPONDING"
        assert req.note == ""

    def test_empty_updated_by_raises(self):
        with pytest.raises(ValidationError):
            TransitionRequest(new_status="RESPONDING", updated_by="")

    def test_negative_area_raises(self):
        with pytest.raises(ValidationError):
            TransitionRequest(
                new_status="RESPONDING",
                updated_by="sys",
                affected_area_ha=-1.0,
            )

    def test_risk_score_out_of_range(self):
        with pytest.raises(ValidationError):
            TransitionRequest(
                new_status="RESPONDING",
                updated_by="sys",
                risk_score=1.5,
            )

    def test_note_max_length_500(self):
        # Exactly 500 chars is fine
        req = TransitionRequest(new_status="RESPONDING", updated_by="sys", note="x" * 500)
        assert len(req.note) == 500

    def test_note_over_500_raises(self):
        with pytest.raises(ValidationError):
            TransitionRequest(new_status="RESPONDING", updated_by="sys", note="x" * 501)


# ---------------------------------------------------------------------------
# FWIComponentsSchema
# ---------------------------------------------------------------------------


class TestFWIComponentsSchema:
    def test_valid(self):
        fwi = FWIComponentsSchema(
            ffmc=88.5, dmc=12.3, dc=45.0, isi=8.7, bui=10.2, fwi=15.4, dsr=1.2
        )
        assert fwi.fwi == pytest.approx(15.4)


# ---------------------------------------------------------------------------
# SpreadHorizonSchema
# ---------------------------------------------------------------------------


class TestSpreadHorizonSchema:
    def test_valid(self):
        s = SpreadHorizonSchema(
            time_horizon_minutes=30,
            affected_area_hectares=12.5,
            num_active_fire_cells=100,
            num_distinct_fire_regions=1,
            spot_fire_risk=0.15,
            evacuation_buffer_cells=67,
            perimeter_point_count=42,
        )
        assert s.spot_fire_risk == pytest.approx(0.15)

    def test_spot_fire_risk_out_of_range(self):
        with pytest.raises(ValidationError):
            SpreadHorizonSchema(
                time_horizon_minutes=30,
                affected_area_hectares=10.0,
                num_active_fire_cells=50,
                num_distinct_fire_regions=1,
                spot_fire_risk=1.5,
                evacuation_buffer_cells=67,
                perimeter_point_count=20,
            )


# ---------------------------------------------------------------------------
# StreamFrame
# ---------------------------------------------------------------------------


class TestStreamFrame:
    def test_valid(self):
        sf = StreamFrame(
            frame_id=1,
            camera_id="cam-001",
            has_fire=True,
            has_smoke=False,
            max_confidence=0.87,
            aggregate_severity="HIGH",
            detection_count=2,
            inference_time_ms=45.3,
            timestamp=time.time(),
        )
        assert sf.alert_id is None

    def test_with_alert_id(self):
        sf = StreamFrame(
            frame_id=2,
            camera_id="cam-002",
            has_fire=False,
            has_smoke=False,
            max_confidence=0.0,
            aggregate_severity="low",
            detection_count=0,
            inference_time_ms=10.0,
            alert_id="some-uuid",
            timestamp=time.time(),
        )
        assert sf.alert_id == "some-uuid"


# ---------------------------------------------------------------------------
# StreamSubscribeMessage
# ---------------------------------------------------------------------------


class TestStreamSubscribeMessage:
    def test_valid_subscribe(self):
        msg = StreamSubscribeMessage(action="subscribe", camera_id="cam-001")
        assert msg.action == "subscribe"
        assert msg.include_detections is True

    def test_valid_unsubscribe(self):
        msg = StreamSubscribeMessage(action="unsubscribe", camera_id="cam-001")
        assert msg.action == "unsubscribe"

    def test_valid_ping(self):
        msg = StreamSubscribeMessage(action="ping", camera_id="cam-001")
        assert msg.action == "ping"

    def test_invalid_action_raises(self):
        with pytest.raises(ValidationError):
            StreamSubscribeMessage(action="invalid", camera_id="cam-001")


# ---------------------------------------------------------------------------
# HealthResponse / ComponentHealth
# ---------------------------------------------------------------------------


class TestComponentHealth:
    def test_ok_status(self):
        ch = ComponentHealth(status="ok")
        assert ch.status == "ok"
        assert ch.latency_ms is None

    def test_with_latency(self):
        ch = ComponentHealth(status="degraded", latency_ms=120.5, detail="Slow response")
        assert ch.latency_ms == pytest.approx(120.5)


class TestHealthResponse:
    def test_valid(self):
        hr = HealthResponse(
            status="ok",
            version="1.0.0",
            uptime_seconds=3600.0,
            components={
                "detector": ComponentHealth(status="ok"),
                "weather": ComponentHealth(status="ok", latency_ms=50.0),
            },
            active_alerts=2,
            active_incidents=1,
            timestamp=time.time(),
        )
        assert hr.active_alerts == 2
        assert "detector" in hr.components
