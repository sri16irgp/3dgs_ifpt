"""The three color channels independently expose three Gaussian contributions."""
import unittest
import math
try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA PyTorch required')
class ContributorTests(unittest.TestCase):
    def setUp(self):
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer, rasterize_contributors
        from utils.graphics_utils import getProjectionMatrix
        self.contributors = rasterize_contributors
        self.settings = GaussianRasterizationSettings(
            image_height=32, image_width=32, tanfovx=1., tanfovy=1., bg=torch.zeros(3, device='cuda'),
            scale_modifier=1., viewmatrix=torch.eye(4, device='cuda'),
            projmatrix=getProjectionMatrix(.01, 100, math.pi/2, math.pi/2).T.cuda(),
            sh_degree=0, campos=torch.zeros(3, device='cuda'), prefiltered=False, debug=False, antialiasing=False)
        self.rasterizer = GaussianRasterizer(self.settings)
        self.means = torch.tensor([[0, 0, 2], [.15, 0, 2.4], [-.1, .05, 3]], device='cuda', dtype=torch.float32)
        self.opacity = torch.tensor([[.15], [.9], [.99]], device='cuda')
        self.scales = torch.full((3, 3), .25, device='cuda')
        self.rotations = torch.tensor([[1., 0, 0, 0]] * 3, device='cuda')

    def test_argmax_matches_individual_color_weights_and_backward(self):
        self.means.requires_grad_()
        self.opacity.requires_grad_()
        color, _, _ = self.rasterizer(self.means, torch.zeros_like(self.means), self.opacity,
            colors_precomp=torch.eye(3, device='cuda'), scales=self.scales, rotations=self.rotations)
        weights, expected = color.max(0)
        expected[weights == 0] = -1
        ids, result_weights = self.contributors(self.means, self.opacity, self.scales, self.rotations, self.settings)
        torch.testing.assert_close(ids.long(), expected)
        torch.testing.assert_close(result_weights, weights)
        self.assertEqual(int(ids[16, 16]), 1)  # frontmost is not strongest
        color.sum().backward()
        self.assertTrue(torch.isfinite(self.means.grad).all())
        self.assertTrue(torch.isfinite(self.opacity.grad).all())

    def test_opacity_cutoff_and_early_termination(self):
        for values in ([.001, .002, .003], [.99, .99, .99]):
            opacity = torch.tensor(values, device='cuda')[:, None]
            color, _, _ = self.rasterizer(self.means, torch.zeros_like(self.means), opacity,
                colors_precomp=torch.eye(3, device='cuda'), scales=self.scales, rotations=self.rotations)
            weights, expected = color.max(0)
            expected[weights == 0] = -1
            actual = self.contributors(self.means, opacity, self.scales, self.rotations, self.settings)
            torch.testing.assert_close(actual[0].long(), expected)
            torch.testing.assert_close(actual[1], weights)

    def test_empty_and_culled(self):
        for n in (0, 3):
            means = self.means[:n].clone()
            means[:, 2] = -2
            ids, weights = self.contributors(means, self.opacity[:n], self.scales[:n], self.rotations[:n], self.settings)
            self.assertTrue((ids == -1).all())
            self.assertTrue((weights == 0).all())

    def test_covariance_antialiasing_and_nondefault_stream(self):
        from utils.general_utils import strip_symmetric, build_scaling_rotation
        transform = build_scaling_rotation(self.scales, self.rotations)
        covariance = strip_symmetric(transform @ transform.transpose(1, 2))
        settings = self.settings._replace(antialiasing=True)
        expected = self.contributors(self.means, self.opacity, self.scales, self.rotations, settings)
        with torch.cuda.stream(torch.cuda.Stream()):
            actual = self.contributors(self.means, self.opacity, self.scales, self.rotations, settings, covariance)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])


if __name__ == '__main__':
    unittest.main()
