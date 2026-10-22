import os
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
