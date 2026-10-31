import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from motion2mixamorig import youtube


class YoutubeProxyTests(unittest.TestCase):
    def test_default_download_uses_tor(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "clip.mp4"
            video.touch()
            with patch.dict(os.environ, {}, clear=True), patch.object(
                youtube, "_yt_dlp_binary", return_value="yt-dlp"
            ), patch.object(
                youtube, "_ensure_tor_proxy", return_value=youtube.DEFAULT_TOR_PROXY
            ) as tor, patch.object(
                youtube.subprocess, "run",
                return_value=SimpleNamespace(returncode=0, stdout=str(video), stderr=""),
            ) as runner, redirect_stdout(io.StringIO()):
                self.assertEqual(
                    youtube.download_youtube_video("https://example.com/clip", Path(tmp)),
                    video,
                )
            tor.assert_called_once()
            cmd = runner.call_args.args[0]
            self.assertEqual(cmd[cmd.index("--proxy") + 1], youtube.DEFAULT_TOR_PROXY)
            runner.assert_called_once()

    def test_unavailable_or_failed_tor_does_not_fall_back_to_direct(self):
        for available in (False, True):
            with self.subTest(available=available), tempfile.TemporaryDirectory() as tmp:
                with patch.dict(os.environ, {}, clear=True), patch.object(
                    youtube, "_yt_dlp_binary", return_value="yt-dlp"
                ), patch.object(
                    youtube, "_ensure_tor_proxy",
                    return_value=youtube.DEFAULT_TOR_PROXY if available else None,
                ), patch.object(
                    youtube.subprocess, "run",
                    return_value=SimpleNamespace(returncode=1, stdout="", stderr="failed"),
                ) as runner, redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(
                        RuntimeError, "Failed to download" if available else "Tor proxy is unavailable"
                    ):
                        youtube.download_youtube_video("https://example.com/clip", Path(tmp))
                self.assertEqual(runner.call_count, 1 if available else 0)
                if available:
                    cmd = runner.call_args.args[0]
                    self.assertEqual(cmd[cmd.index("--proxy") + 1], youtube.DEFAULT_TOR_PROXY)

    def test_explicit_and_environment_proxy_overrides(self):
        cases = [
            ("http://localhost:8080", {"YT_DLP_PROXY": "http://localhost:9000"}, "http://localhost:8080"),
            ("", {"YT_DLP_PROXY": "http://localhost:9000"}, ""),
            (None, {"YT_DLP_PROXY": "http://localhost:9000"}, "http://localhost:9000"),
            (None, {"YT_DLP_PROXY": ""}, ""),
        ]
        for proxy, env, expected in cases:
            with self.subTest(proxy=proxy, env=env), tempfile.TemporaryDirectory() as tmp:
                video = Path(tmp) / "clip.mp4"
                video.touch()
                with patch.dict(os.environ, env, clear=True), patch.object(
                    youtube, "_yt_dlp_binary", return_value="yt-dlp"
                ), patch.object(youtube, "_ensure_tor_proxy") as tor, patch.object(
                    youtube.subprocess, "run",
                    return_value=SimpleNamespace(returncode=0, stdout=str(video), stderr=""),
                ) as runner, redirect_stdout(io.StringIO()):
                    youtube.download_youtube_video(
                        "https://example.com/clip", Path(tmp), proxy=proxy
                    )
                tor.assert_not_called()
                cmd = runner.call_args.args[0]
                self.assertEqual(cmd[cmd.index("--proxy") + 1], expected)

    def test_cli_preserves_explicit_direct_option(self):
        with patch.object(youtube, "run_youtube_to_bvh") as run:
            youtube.main(["clip.mp4", "--proxy", ""])
        self.assertEqual(run.call_args.kwargs["proxy"], "")


if __name__ == "__main__":
    unittest.main()
