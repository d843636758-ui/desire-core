from __future__ import annotations

import copy
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, patch

os.environ.setdefault("DATA_DIR", "/tmp/desire-core-tests")
os.environ.setdefault("PHOSPHENE_ENABLED", "false")

import server
from desire_core import DesireState, JSONStore


class PhosphenePureHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime.fromisoformat(
            "2026-07-20T10:00:00+08:00"
        )

    def test_hidden_surprise_is_not_visible_before_reveal(self) -> None:
        task = {
            "id": "task_hidden",
            "title": "以后再揭晓",
            "status": "pending",
            "type": "surprise",
            "revealMode": "next_visit",
            "revealedAt": None,
        }

        self.assertFalse(
            server.phosphene_task_visible(
                task,
                now=self.now,
            )
        )

        task["revealedAt"] = (
            "2026-07-20T09:59:00+08:00"
        )

        self.assertTrue(
            server.phosphene_task_visible(
                task,
                now=self.now,
            )
        )

    def test_deadline_increases_task_intensity(self) -> None:
        task = {
            "status": "pending",
            "difficulty": "easy",
            "deadline": "2026-07-20T11:00:00+08:00",
        }

        self.assertEqual(
            server.phosphene_task_intensity(
                task,
                now=self.now,
            ),
            0.88,
        )

    def test_ingestion_is_deduplicated(self) -> None:
        task = {
            "id": "task_1",
            "title": "喝水",
            "description": "完成一杯水",
            "status": "pending",
            "type": "daily",
            "difficulty": "easy",
            "deadline": "2026-07-20T11:00:00+08:00",
            "occurrence_date": "2026-07-20",
        }

        state, first_added = (
            server.ingest_phosphene_tasks(
                DesireState(),
                [task],
                now_value=self.now.isoformat(),
            )
        )
        state, second_added = (
            server.ingest_phosphene_tasks(
                state,
                [task],
                now_value=self.now.isoformat(),
            )
        )

        self.assertEqual(first_added, 1)
        self.assertEqual(second_added, 0)
        self.assertTrue(
            any(
                interest.source_counts.get(
                    "phosphene"
                ) == 1
                for interest in state.interests
            )
        )

    def test_attention_marks_in_session_followup(self) -> None:
        attention = server.phosphene_attention(
            {
                "queues": {
                    "pending_tasks": 2,
                    "awaiting_review": 1,
                    "pending_redemptions": 0,
                }
            },
            [
                {
                    "title": "复习十分钟",
                    "status": "pending",
                }
            ],
        )

        self.assertTrue(
            attention[
                "autonomous_followup_eligible"
            ]
        )
        self.assertEqual(
            attention["suggested_task"],
            "复习十分钟",
        )


class PhospheneSyncTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.original_store = server.STORE
        self.original_values = {
            "PHOSPHENE_ENABLED":
                server.PHOSPHENE_ENABLED,
            "PHOSPHENE_MCP_URL":
                server.PHOSPHENE_MCP_URL,
            "PHOSPHENE_MCP_TOKEN":
                server.PHOSPHENE_MCP_TOKEN,
            "PHOSPHENE_MODE":
                server.PHOSPHENE_MODE,
            "PHOSPHENE_STATUS":
                copy.deepcopy(
                    server.PHOSPHENE_STATUS
                ),
        }

        server.STORE = JSONStore(
            os.path.join(
                self.tempdir.name,
                "desire_state.json",
            )
        )
        server.PHOSPHENE_ENABLED = True
        server.PHOSPHENE_MCP_URL = (
            "https://example.test/mcp"
        )
        server.PHOSPHENE_MCP_TOKEN = (
            "phosphene_ai_test"
        )
        server.PHOSPHENE_MODE = "shadow"
        server.PHOSPHENE_STATUS.update(
            {
                "enabled": True,
                "configured": True,
                "mode": "shadow",
                "last_ok_at": None,
                "last_error": None,
                "last_attempt_at": None,
                "last_ingested": 0,
                "overview": {},
                "tasks": [],
            }
        )

        self.snapshot = (
            {
                "queues": {
                    "pending_tasks": 1,
                    "awaiting_review": 0,
                    "pending_redemptions": 0,
                },
                "labels": {
                    "user": "user",
                    "ai": "AI",
                },
            },
            {
                "items": [
                    {
                        "id": "task_1",
                        "title": "完成测试任务",
                        "description": "确认桥接工作",
                        "status": "pending",
                        "type": "daily",
                        "difficulty": "medium",
                        "revealMode": "immediate",
                        "deadlineAt": None,
                    }
                ]
            },
            {"items": []},
        )

    async def asyncTearDown(self) -> None:
        server.STORE = self.original_store
        server.PHOSPHENE_ENABLED = (
            self.original_values[
                "PHOSPHENE_ENABLED"
            ]
        )
        server.PHOSPHENE_MCP_URL = (
            self.original_values[
                "PHOSPHENE_MCP_URL"
            ]
        )
        server.PHOSPHENE_MCP_TOKEN = (
            self.original_values[
                "PHOSPHENE_MCP_TOKEN"
            ]
        )
        server.PHOSPHENE_MODE = (
            self.original_values[
                "PHOSPHENE_MODE"
            ]
        )
        server.PHOSPHENE_STATUS.clear()
        server.PHOSPHENE_STATUS.update(
            self.original_values[
                "PHOSPHENE_STATUS"
            ]
        )
        self.tempdir.cleanup()

    async def test_shadow_caches_without_mutating_desire(self) -> None:
        with patch.object(
            server,
            "call_phosphene_snapshot",
            AsyncMock(
                return_value=self.snapshot
            ),
        ):
            await server.sync_from_phosphene(
                force=True
            )

        state = server.STORE.load()

        self.assertEqual(
            server.PHOSPHENE_STATUS[
                "tasks"
            ][0]["title"],
            "完成测试任务",
        )
        self.assertEqual(
            server.PHOSPHENE_STATUS[
                "last_ingested"
            ],
            0,
        )
        self.assertFalse(
            server.PHOSPHENE_STATUS[
                "attention"
            ][
                "autonomous_followup_eligible"
            ]
        )
        self.assertFalse(
            any(
                key.startswith("phosphene:")
                for key in state.upstream_seen
            )
        )

    async def test_in_session_ingests_once(self) -> None:
        server.PHOSPHENE_MODE = "in_session"

        with patch.object(
            server,
            "call_phosphene_snapshot",
            AsyncMock(
                return_value=self.snapshot
            ),
        ):
            await server.sync_from_phosphene(
                force=True
            )
            await server.sync_from_phosphene(
                force=True
            )

        state = server.STORE.load()

        self.assertEqual(
            sum(
                1
                for key in state.upstream_seen
                if key.startswith("phosphene:")
            ),
            1,
        )
        self.assertEqual(
            server.PHOSPHENE_STATUS[
                "last_ingested"
            ],
            0,
        )

        private = server.state_view(
            include_phosphene_details=True
        )["phosphene"]

        self.assertFalse(
            private["attention"][
                "autonomous_followup_eligible"
            ]
        )

        def enable_driven(state):
            state.gates[
                "DESIRE_DRIVEN"
            ] = True
            return state

        server.STORE.update(
            enable_driven
        )

        private = server.state_view(
            include_phosphene_details=True
        )["phosphene"]

        self.assertTrue(
            private["attention"][
                "autonomous_followup_eligible"
            ]
        )

    async def test_public_state_hides_task_details(self) -> None:
        with patch.object(
            server,
            "call_phosphene_snapshot",
            AsyncMock(
                return_value=self.snapshot
            ),
        ):
            await server.sync_from_phosphene(
                force=True
            )

        public = server.state_view(
            include_phosphene_details=False
        )["phosphene"]
        private = server.state_view(
            include_phosphene_details=True
        )["phosphene"]

        self.assertNotIn("tasks", public)
        self.assertIn("tasks", private)


if __name__ == "__main__":
    unittest.main()
