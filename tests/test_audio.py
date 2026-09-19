"""Standalone regression tests: python -m unittest discover -s tests -v.

Load the audio helpers without starting ComfyUI. Torch is mocked because these
tests exercise process errors, metadata, and lazy loading, not tensor math.
"""
import ast
from collections.abc import Mapping
from pathlib import Path
import re
import subprocess
import unittest
from unittest.mock import Mock, patch


def load_helpers():
    path = Path(__file__).resolve().parents[1] / 'videohelpersuite/utils.py'
    tree = ast.parse(path.read_text())
    names = {'get_audio', 'LazyAudioMap', 'lazy_get_audio'}
    tree.body = [node for node in tree.body if getattr(node, 'name', None) in names]
    namespace = dict(Mapping=Mapping, re=re, subprocess=subprocess,
                     torch=Mock(), ffmpeg_path='ffmpeg',
                     ENCODE_ARGS=('utf-8', 'backslashreplace'))
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


class AudioTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_helpers()

    def failure(self, stderr):
        return subprocess.CalledProcessError(1, ['ffmpeg'], stderr=stderr)

    def test_video_only_is_cached_empty_mapping(self):
        error = self.failure(b'Input #0, mov:\n  Stream #0:0[0x1](und): Video: h264\n'
                             b'Output file does not contain any stream\n')
        with patch.object(subprocess, 'run', side_effect=error) as run:
            audio = self.ns['lazy_get_audio']('silent.mp4')
            run.assert_not_called()
            self.assertEqual(list(audio), [])
            self.assertEqual(len(audio), 0)
            self.assertEqual(dict(audio), {})
            with self.assertRaises(KeyError):
                audio['waveform']
            run.assert_called_once()

    def test_real_errors_propagate(self):
        for stderr in [
            b'No such file or directory',
            b'Invalid data found when processing input',
            b'Stream #0:0: Video: h264\nError decoding input',
            b'Output file does not contain any stream',
            b'Stream #0:0: Video: h264\nStream #0:1: Audio: aac\n'
            b'Output file does not contain any stream',
        ]:
            with self.subTest(stderr=stderr):
                error = self.failure(stderr)
                with patch.object(subprocess, 'run', side_effect=error):
                    with self.assertRaisesRegex(Exception, 'VHS failed to extract audio') as ctx:
                        dict(self.ns['lazy_get_audio']('broken.mp4'))
                    self.assertIs(ctx.exception.__cause__, error)
                    self.assertIn(stderr.decode(), str(ctx.exception))

    def test_audio_and_trimming_are_preserved(self):
        for layout, channels in [('mono', 1), ('stereo', 2)]:
            with self.subTest(layout=layout):
                self.ns = load_helpers()
                result = subprocess.CompletedProcess([], 0, b'\x00' * 16,
                    f'Stream #0:1: Audio: aac, 48000 Hz, {layout}, fltp'.encode())
                with patch.object(subprocess, 'run', return_value=result) as run:
                    audio = self.ns['get_audio']('audio.mp4', 2, 3)
                self.assertEqual(audio['sample_rate'], 48000)
                self.ns['torch'].frombuffer.return_value.reshape.assert_called_once_with((-1, channels))
                self.assertEqual(run.call_args.args[0], ['ffmpeg', '-i', 'audio.mp4',
                    '-ss', '2', '-t', '3', '-vn', '-sn', '-dn', '-f', 'f32le', '-'])
                self.assertTrue(run.call_args.kwargs['check'])

    def test_empty_success_does_not_call_frombuffer(self):
        result = subprocess.CompletedProcess([], 0, b'',
                    b'Stream #0:1: Audio: aac, 44100 Hz, stereo, fltp')
        with patch.object(subprocess, 'run', return_value=result):
            audio = self.ns['get_audio']('audio.mp4', 1000)
        self.ns['torch'].frombuffer.assert_not_called()
        self.ns['torch'].empty.assert_called_once_with(0, dtype=self.ns['torch'].float32)
        self.assertEqual(audio['sample_rate'], 44100)


if __name__ == '__main__':
    unittest.main()
