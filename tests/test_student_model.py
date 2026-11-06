import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
HAS_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH, 'PyTorch is required for trainable candidate checks')
class StudentModelTests(unittest.TestCase):
    def test_node_weights_cover_the_exact_mobile_parameter_set(self):
        from student.model import Candidate
        model = Candidate()
        weights = model.ncnn_weights()
        self.assertEqual(len(weights), 26)
        self.assertEqual(len(weights[8]), 8)
        self.assertEqual(sum(value.numel() for group in weights for value in group), 246112)
        self.assertEqual({id(value) for group in weights for value in group},
                         {id(value) for value in model.parameters()})

    def test_backpropagation_reaches_prior_frame_state(self):
        import torch
        from student.model import Candidate
        torch.set_num_threads(2)
        torch.manual_seed(123)
        model = Candidate()
        image = torch.rand(1, 3, 256, 256)
        spatial, identity = model.encode_identity(image)
        audio = torch.randn(1, 8, 40)
        _, state1 = model.forward_frame(audio, torch.zeros(1, 128), spatial, identity)
        state1.retain_grad()
        rgb, state2 = model.forward_frame(audio * .5, state1, spatial, identity)
        (rgb.square().mean() + state2.square().mean()).backward()
        self.assertEqual(tuple(rgb.shape), (1, 3, 256, 256))
        self.assertGreater(float(state1.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_optimizer_checkpoint_and_reexport_preserve_exact_payload(self):
        from scripts.student_export import export
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = export(root / 'first', frames=8, smoke_train=True)
            second = export(root / 'second', checkpoint=root / 'first/checkpoint.pt', frames=8)
            self.assertEqual(first['bundle_sha256'], second['bundle_sha256'])
            self.assertTrue(first['gradient_check']['all_parameter_gradients_finite'])
            self.assertGreater(first['gradient_check']['updated_parameter_tensors'], 0)
            self.assertFalse(first['provenance']['teacher_trained'])
            self.assertFalse(first['video_quality_validated'])


@unittest.skipUnless(HAS_TORCH, 'PyTorch is required for reference candidate checks')
class ReferenceCandidateTests(unittest.TestCase):
    def test_zero_head_preserves_batched_reference_exactly(self):
        import torch
        from student.reference_model import ReferenceCandidate
        torch.set_num_threads(2)
        model = ReferenceCandidate()
        image = torch.rand(2, 3, 256, 256)
        with torch.no_grad():
            output, state = model(image, torch.randn(2, 8, 40), torch.zeros(2, 128))
        torch.testing.assert_close(output, image, rtol=0, atol=0)
        self.assertEqual(tuple(state.shape), (2, 128))

    def test_recurrent_state_is_bounded_on_a_longer_sequence(self):
        import torch
        from student.reference_model import ReferenceCandidate
        torch.set_num_threads(2)
        model = ReferenceCandidate()
        with torch.inference_mode():
            cache, identity = model.encode_identity(torch.rand(1, 3, 256, 256))
            state = torch.zeros(1, 128)
            for _ in range(32):
                rgb, state = model.forward_frame(torch.randn(1, 8, 40)*20, state, cache, identity)
                self.assertTrue(torch.isfinite(rgb).all())
                self.assertLessEqual(float(state.abs().max()), 1.)

    def test_warp_and_residual_head_can_receive_gradients(self):
        import torch
        from student.reference_model import ReferenceCandidate
        torch.set_num_threads(2)
        model = ReferenceCandidate()
        image = torch.rand(1, 3, 256, 256)
        rgb, state = model(image, torch.randn(1, 8, 40), torch.zeros(1, 128))
        ((rgb-image.roll(2, dims=-1)).square().mean()+state.square().mean()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(float(model.output.weight.grad[:2].abs().sum()), 0.)
        self.assertGreater(float(model.output.weight.grad[2:].abs().sum()), 0.)

    def test_v0_exporter_rejects_v1_before_creating_an_output(self):
        import torch
        from student.reference_model import ReferenceCandidate, ARCHITECTURE
        from scripts.student_export import export
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / 'v1.pt'
            torch.save({'schema_version': 1, 'architecture': ARCHITECTURE,
                        'state_dict': ReferenceCandidate().state_dict(), 'provenance': {}}, checkpoint)
            with self.assertRaisesRegex(ValueError, 'supports recurrent_conditioned_rgb_v0 only'):
                export(root / 'export', checkpoint=checkpoint)
            self.assertFalse((root / 'export').exists())


@unittest.skipUnless(HAS_TORCH, 'PyTorch required for mouth refinement checks')
class MouthRefinementTests(unittest.TestCase):
    def test_letterbox_policy_only_masks_verified_padding(self):
        import numpy as np
        import torch
        from PIL import Image
        from student.data import square_image, sha256
        from student.mobile import fixed_letterbox_mask, FrameDecoder
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'reference.png';Image.new('RGB',(48,64),(160,130,120)).save(source)
            with Image.open(source) as image:reference=square_image(image)
            data={'reference':reference,'frames':np.repeat(reference[None],8,axis=0)}
            record={'source_reference':str(source),'source_reference_sha256':sha256(source)}
            mask,policy=fixed_letterbox_mask(record,data)
            self.assertEqual(policy['content_box'],[32,0,224,256])
            class Dummy(torch.nn.Module):
                def forward_frame(self,audio,state,cache,identity):return torch.full_like(cache[1],.5),state
            image=torch.from_numpy(reference).permute(2,0,1)[None].float()/255
            rgb,state=FrameDecoder(Dummy(),mask)(image,torch.zeros(1,32,64,64),torch.zeros(1,128),torch.zeros(1,8,40),torch.zeros(1,128))
            self.assertTrue(torch.all(rgb[:,:,:,:32]==0))
            self.assertTrue(torch.all(rgb[:,:,:,32:224]==.5))
            data['frames'][0,10,0,0]=1
            with self.assertRaisesRegex(ValueError,'non-padding'):fixed_letterbox_mask(record,data)

    def test_initial_identity_and_nonzero_correction_stays_inside_patch(self):
        import torch
        from student.mouth_model import MouthRefinement, PATCH_BOX
        torch.set_num_threads(2)
        head = MouthRefinement()
        base = torch.rand(2, 3, 256, 256)
        inputs = (base, torch.rand_like(base), torch.randn(2, 8, 40), torch.randn(2, 128))
        with torch.no_grad():
            torch.testing.assert_close(head(*inputs), base, rtol=0, atol=0)
            head.output.bias.fill_(.2)
            changed = head(*inputs)
        x0, y0, x1, y1 = PATCH_BOX
        outside = torch.ones(256, 256, dtype=torch.bool); outside[y0:y1, x0:x1] = False
        torch.testing.assert_close(changed[:, :, outside], base[:, :, outside], rtol=0, atol=0)
        self.assertGreater(float((changed-base).abs().max()), .1)

    def test_audio_and_state_receive_gradients_through_local_decoder(self):
        import torch
        from student.mouth_model import MouthRefinement
        torch.set_num_threads(2)
        head = MouthRefinement()
        torch.nn.init.normal_(head.output.weight, std=.02)
        audio = torch.randn(1, 8, 40, requires_grad=True)
        state = torch.randn(1, 128, requires_grad=True)
        head(torch.full((1, 3, 256, 256), .5), torch.rand(1, 3, 256, 256), audio, state).mean().backward()
        self.assertGreater(float(audio.grad.abs().sum()), 0.)
        self.assertGreater(float(state.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_dark_area_proxy_distinguishes_open_cavity_from_closed_seam(self):
        import torch
        from student.mouth_model import dark_area_proxy
        images = torch.full((2, 3, 256, 256), .6)
        images[0, :, 117:118, 116:142] = .05
        images[1, :, 111:124, 116:142] = .05
        scores = dark_area_proxy(images)
        self.assertGreater(float(scores[1]), float(scores[0])*5)


if __name__ == '__main__':
    unittest.main()
