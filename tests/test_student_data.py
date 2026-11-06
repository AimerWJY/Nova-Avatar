import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from student.audio import features, histories

HAS_AV = importlib.util.find_spec('av') is not None
HAS_TORCH = importlib.util.find_spec('torch') is not None


class AudioContractTests(unittest.TestCase):
    def test_silence_and_future_samples_do_not_change_causal_features(self):
        silence = features(np.zeros(16000, dtype=np.float32), [0., .5])
        np.testing.assert_array_equal(silence, np.full((2, 40), -1, dtype=np.float32))
        waveform = np.sin(np.arange(16000)*2*np.pi*400/16000).astype(np.float32)
        original = features(waveform, [0., .25, .5])
        waveform[8000:] = 1
        np.testing.assert_array_equal(features(waveform, [0., .25, .5]), original)
        self.assertGreater(float(original[1].std()), .01)

    def test_history_padding_never_reads_future_frames(self):
        source = np.arange(10*40, dtype=np.float32).reshape(10, 40)
        history = histories(source)
        np.testing.assert_array_equal(history[0], np.repeat(source[:1], 8, axis=0))
        np.testing.assert_array_equal(history[9], source[2:10])
        np.testing.assert_array_equal(history[3][-1], source[3])


@unittest.skipUnless(HAS_AV, 'PyAV is required for teacher media decoding')
class TeacherDataTests(unittest.TestCase):
    def create_pair(self, root):
        from scripts.student_train import write_video
        from student.data import prepare
        frames = np.zeros((8, 256, 256, 3), dtype=np.uint8)
        for i in range(8):
            frames[i, 70:180, 70+i:180+i] = [100, 150, 200]
        video, reference = root / 'source.mp4', root / 'reference.png'
        write_video(video, frames, (.1*np.sin(np.arange(8000)*2*np.pi*400/16000)).astype(np.float32))
        Image.fromarray(frames[0]).save(reference)
        manifest = root / 'prepared/manifest.json'
        prepare(video, reference, manifest.parent, 'test_person', 'synthetic unit-test media')
        return manifest

    def test_decode_alignment_integrity_and_review_gate(self):
        from student.data import load
        with tempfile.TemporaryDirectory() as folder:
            manifest = self.create_pair(Path(folder))
            record, data = load(manifest)
            self.assertEqual(record['frames'], 8)
            self.assertEqual(data['audio_features'].shape, (8, 40))
            np.testing.assert_allclose(data['times'], np.arange(8)/16)
            record['review']['status'] = 'pending'
            manifest.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'unreviewed'):
                load(manifest)
            record['review']['status'] = 'approved'
            manifest.write_text(json.dumps(record))
            (manifest.parent / 'sample.npz').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                load(manifest)

    @unittest.skipUnless(HAS_TORCH, 'PyTorch required for real-pair training integration')
    def test_one_step_training_keeps_diagnostic_scope_and_writes_aligned_preview(self):
        import av
        import torch
        from scripts.student_train import train
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = self.create_pair(root)
            args = Namespace(dataset=manifest, output=root / 'training', init_checkpoint=None,
                             steps=1, sequence_length=2, lr=.001, seed=713, device='cpu',
                             mouth_box=[.35, .36, .65, .52])
            report = train(args)
            self.assertFalse(report['independent_validation'])
            self.assertFalse(report['lip_sync_verified'])
            self.assertFalse(report['video_quality_validated'])
            checkpoint = torch.load(args.output / 'checkpoint.pt', weights_only=True)
            self.assertTrue(checkpoint['provenance']['teacher_trained'])
            with av.open(str(args.output / 'comparison.mp4')) as container:
                self.assertTrue(container.streams.audio)
                frames = list(container.decode(video=0))
                self.assertEqual(len(frames), 8)
                self.assertEqual((frames[0].width, frames[0].height), (512, 256))

    @unittest.skipUnless(HAS_TORCH, 'PyTorch required for architecture comparison')
    def test_comparison_separates_reference_copy_from_quality_validation(self):
        import torch
        from student.data import load, sha256
        from student.model import Candidate, ARCHITECTURE
        from student.reference_model import ReferenceCandidate, ARCHITECTURE as REFERENCE_ARCHITECTURE
        from scripts.student_compare import compare
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); manifest = self.create_pair(root)
            record, _ = load(manifest)
            for label, constructor, architecture in [('v0', Candidate, ARCHITECTURE),
                                                      ('v1', ReferenceCandidate, REFERENCE_ARCHITECTURE)]:
                torch.save({'schema_version': 1, 'architecture': architecture,
                            'state_dict': constructor().state_dict(),
                            'provenance': {'source_video_sha256': record['source_video_sha256'],
                                           'dataset_manifest_sha256': sha256(manifest)}}, root / (label+'.pt'))
            report = compare(manifest, root / 'v0.pt', root / 'v1.pt', root / 'comparison')
            self.assertAlmostEqual(report['v1']['normal_audio']['mae'], report['static_reference_copy']['mae'], places=6)
            self.assertFalse(report['independent_validation'])
            self.assertFalse(report['video_quality_validated'])
            self.assertFalse(report['phone_v1_validated'])

    @unittest.skipUnless(HAS_TORCH, 'PyTorch required for local mouth training')
    def test_local_mouth_training_keeps_frozen_backbone_and_diagnostic_scope(self):
        import torch
        from student.data import sha256
        from student.reference_model import ReferenceCandidate, ARCHITECTURE
        from scripts.student_mouth_train import train
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); manifest = self.create_pair(root)
            checkpoint = root / 'base.pt'
            torch.save({'schema_version': 1, 'architecture': ARCHITECTURE,
                        'state_dict': ReferenceCandidate().state_dict(),
                        'provenance': {'dataset_manifest_sha256': sha256(manifest)}}, checkpoint)
            report = train(Namespace(dataset=manifest, base_checkpoint=checkpoint, output=root / 'mouth',
                                     steps=2, batch_size=2, lr=.001, seed=713, device='cpu'))
            self.assertTrue(report['base_frozen_and_unchanged'])
            self.assertEqual(report['outside_patch_max_error'], 0)
            self.assertFalse(report['independent_validation'])
            self.assertFalse(report['lip_sync_verified'])
            self.assertFalse(report['phone_validated'])
            self.assertFalse(report['video_quality_validated'])

    @unittest.skipUnless(HAS_TORCH, 'PyTorch required for independent audio preview')
    def test_independent_audio_preview_rejects_training_audio_and_keeps_scope(self):
        import torch
        import wave
        from student.data import sha256
        from student.mouth_model import MouthCandidate, ARCHITECTURE
        from scripts.student_infer import infer
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); manifest = self.create_pair(root); checkpoint = root/'v2.pt'
            torch.save({'schema_version': 1, 'architecture': ARCHITECTURE,
                        'state_dict': MouthCandidate().state_dict(),
                        'provenance': {'dataset_manifest_sha256': sha256(manifest)}}, checkpoint)
            args = Namespace(checkpoint=checkpoint, training_manifest=manifest, audio_source=root/'source.mp4',
                             reference=root/'reference.png', output=root/'preview', max_seconds=.5)
            with self.assertRaisesRegex(ValueError, 'duplicates'):
                infer(args)
            self.assertFalse(args.output.exists())
            args.audio_source = root/'independent.wav'
            with wave.open(str(args.audio_source), 'wb') as stream:
                stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
                stream.writeframes((np.sin(np.arange(8000)*2*np.pi*880/16000)*3000).astype('<i2').tobytes())
            report = infer(args)
            self.assertTrue(report['all_frames_finite'])
            self.assertFalse(report['lip_sync_verified'])
            self.assertFalse(report['generalization_verified'])
            self.assertFalse(report['phone_validated'])
            self.assertFalse(report['video_quality_validated'])

    @unittest.skipUnless(HAS_TORCH, 'PyTorch required for mobile graph parity')
    def test_v2_torchscript_serialization_preserves_sequence_and_rejects_wrong_dataset(self):
        import torch
        from student.data import sha256
        from student.mouth_model import MouthCandidate, ARCHITECTURE
        from scripts.student_mobile_export import export
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); manifest = self.create_pair(root); checkpoint = root/'v2.pt'
            torch.save({'schema_version': 1, 'architecture': ARCHITECTURE,
                        'state_dict': MouthCandidate().state_dict(),
                        'provenance': {'dataset_manifest_sha256': sha256(manifest)}}, checkpoint)
            report = export(checkpoint, manifest, root/'export')
            self.assertTrue(report['torchscript_parity_passed'])
            self.assertEqual(report['reference_frames'], 8)
            self.assertFalse(report['ncnn_conversion']['passed'])
            self.assertFalse(report['phone_validated'])
            changed = json.loads(manifest.read_text()); changed['review']['basis'] = 'different manifest'
            manifest.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'exact prepared dataset'):
                export(checkpoint, manifest, root/'wrong')
            self.assertFalse((root/'wrong').exists())


if __name__ == '__main__':
    unittest.main()
