import os
from pathlib import Path
import unittest

os.environ.setdefault(
    "PHOSPHENE_ENABLED",
    "false",
)

import server


class DeploymentDefaultsTests(unittest.TestCase):
    def test_public_host_defaults_to_aliyun_domain(self):
        self.assertEqual(
            server.PUBLIC_HOST,
            "desire.xunzhou.space",
        )

    def test_legacy_host_remains_allowed_during_migration(self):
        self.assertIn(
            "desirecore.zeabur.app",
            server.PUBLIC_HOSTS,
        )
        self.assertIn(
            "desire.xunzhou.space",
            server.PUBLIC_HOSTS,
        )

    def test_aliyun_script_protects_state_and_mcp(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "deploy-aliyun.sh"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "ALLOW_EMPTY_DESIRE_DATA:-0",
            script,
        )
        self.assertIn(
            "$DATA_DIR/desire_state.json",
            script,
        )
        self.assertIn(
            "DESIRE_MCP_PATH_TOKEN",
            script,
        )
        self.assertIn(
            "handle /mcp*",
            script,
        )
        self.assertIn(
            "respond 404",
            script,
        )
        self.assertIn(
            "caddy validate",
            script,
        )
