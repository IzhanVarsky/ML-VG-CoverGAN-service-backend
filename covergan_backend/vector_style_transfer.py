import os

import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import copy
import pydiffvg
import time
import warnings
import tempfile

warnings.filterwarnings("ignore")
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = pydiffvg.get_device()


# Функция дял загрузки изображения стиля
def image_loader(image, canvas_width, canvas_height):
    # Превращение в тензор
    loader = transforms.ToTensor()
    # Убираем прозрачность и добавляем новое измерение для батча
    image = Image.open(image)
    image = image.resize((canvas_width, canvas_height))
    # Убираем прозрачность и добавляем новое измерение для батча
    image = loader(image)[:3, :, :].unsqueeze(0)
    return image.to(device, torch.float)


def get_contours(canvas_width, canvas_height, shapes, shape_groups):
    shape_groups_new = copy.deepcopy(shape_groups)
    # получаем контур изображения
    with torch.no_grad():
        for shape_group in shape_groups_new:
            if isinstance(shape_group.fill_color, torch.Tensor):
                shape_group.fill_color[0] = 0
                shape_group.fill_color[1] = 0
                shape_group.fill_color[2] = 0
                strike_color = torch.Tensor([1., 1., 1., 1.])
                shape_group.stroke_color = strike_color
            elif isinstance(shape_group.fill_color, pydiffvg.LinearGradient):
                shape_group.fill_color.begin = 0
                shape_group.fill_color.end = 0
                shape_group.fill_color.stop_colors = 0
                begin = torch.Tensor(1.)
                end = torch.Tensor(1.)
                stop_colors = torch.Tensor(1.)
                shape_group.stroke_color.begin = begin
                shape_group.stroke_color.end = end
                shape_group.stroke_color.stop_colors = stop_colors

        # Растеризуем исходную картинку по параметрам объектов
        scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height,
                                                             shapes, shape_groups_new)
        render = pydiffvg.RenderFunction.apply
        img = render(canvas_width,
                     canvas_height,
                     2,
                     2,
                     0,
                     None,
                     *scene_args)
    return img[:, :, :3].unsqueeze(0).permute(0, 3, 1, 2)


# Функция переноса стиля
def transfer_style_vector_image(content_img, style_img,
                                canvas_width, canvas_height,
                                shapes, shape_groups,
                                num_iter, point_rate,
                                color_rate, width_rate):
    # Изображения стиля
    style_img = image_loader(style_img, canvas_width, canvas_height)

    content_img = content_img[:, :, :3].unsqueeze(0).permute(0, 3, 1, 2).to(device)

    # Проверяем, чтобы размеры совпадали
    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    # Класс потери изломанности линий
    class LineLoss(nn.Module):
        def __init__(self, target_contour):
            super(LineLoss, self).__init__()
            self.target_contour = target_contour.detach()
            dim1 = int(self.target_contour.size()[2] * 0.25)
            dim2 = int(self.target_contour.size()[3] * 0.25)
            self.transform = transforms.RandomCrop(size=(dim1, dim2))

        def forward(self, input_contour):
            img1 = self.transform(self.target_contour)
            img2 = self.transform(input_contour)
            self.loss = F.l1_loss(img1, img2)
            return self.loss

    class LPIPS(torch.nn.Module):
        def __init__(self, pretrained=True, normalize=False, pre_relu=True):
            """
            Args:
                pre_relu(bool): if True, selects features **before** reLU activations
            """
            super(LPIPS, self).__init__()
            # VGG using perceptually-learned weights (LPIPS metric)
            self.normalize = normalize
            self.pretrained = pretrained

            self.feature_extractor = LPIPS._FeatureExtractor(pretrained, pre_relu)

        def _l2_normalize_features(self, x, eps=1e-10):
            nrm = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True))
            return x / (nrm + eps)

        def sample_xform(self):
            color_scale = torch.rand(size=(3,)) * 2.0
            return dict(color_scale=color_scale)

        def xform(self, im, params):
            color_scale = params["color_scale"].view(1, 3, 1, 1).to(im.device)
            im = im * color_scale

            return im

        def forward(self, pred, target):
            """Compare VGG features of two inputs."""

            p = self.sample_xform()
            pred = self.xform(pred, p)
            target = self.xform(target, p)

            # Get VGG features
            pred = self.feature_extractor(pred)
            target = self.feature_extractor(target)

            # у нас стоит self.normalize=True и self.pre_relu=True

            # L2 normalize features
            # if self.normalize:
            pred = [self._l2_normalize_features(f) for f in pred]
            target = [self._l2_normalize_features(f) for f in target]

            # TODO(mgharbi) Apply Richard's linear weights?

            if self.normalize:
                diffs = [torch.sum((p - t) ** 2, 1) for (p, t) in zip(pred, target)]
            else:
                # mean instead of sum to avoid super high range
                diffs = [torch.mean((p - t) ** 2, 1) for (p, t) in zip(pred, target)]

            # Spatial average
            diffs = [diff.mean([1, 2]) for diff in diffs]

            return sum(diffs).mean(0)

        class _FeatureExtractor(torch.nn.Module):
            def __init__(self, pretrained, pre_relu):
                super(LPIPS._FeatureExtractor, self).__init__()
                vgg_pretrained = models.vgg19(pretrained=pretrained).features.eval()
                vgg_pretrained.to(pydiffvg.get_device())

                self.breakpoints = [0, 4, 9, 16, 23, 30, 36]
                self.weights = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
                if pre_relu:
                    for i, _ in enumerate(self.breakpoints[1:]):
                        self.breakpoints[i + 1] -= 1

                k = 0
                # Разбиваем по relu
                for i, b in enumerate(self.breakpoints[:-1]):
                    ops = torch.nn.Sequential()
                    for idx in range(b, self.breakpoints[i + 1]):
                        op = vgg_pretrained[idx]
                        ops.add_module(str(idx), op)

                    self.add_module("group{}".format(i), ops)

                # Нормализация
                self.register_buffer("shift", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                self.register_buffer("scale", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

            def forward(self, x):
                feats = []
                x = (x - self.shift) / self.scale
                for idx in range(len(self.breakpoints) - 1):
                    m = getattr(self, "group{}".format(idx))
                    x = m(x) * self.weights[idx]
                    feats.append(x)
                return feats

    # Функция переноса стиля
    def run_style_transfer(content_img, style_img,
                           canvas_width, canvas_height,
                           shapes, shape_groups,
                           num_steps, point_rate, color_rate, width_rate):

        """Run the style transfer."""
        print('Building the style transfer model..')

        # берем контур content_img
        contour = get_contours(canvas_width, canvas_height, shapes, shape_groups)

        pydiffvg.imwrite(contour.squeeze(0).permute(1, 2, 0).cpu(), 'contour.png', gamma=1.0)

        # Берем нашу модель и считаем content_loss и style_loss
        perception_loss = LPIPS().to(pydiffvg.get_device())
        line_loss = LineLoss(contour)

        # Списки с параметрами форм векторного изображения:
        # опорные точки, цвет, длина штрихов
        point_params = []
        color_params = []
        stroke_width_params = []

        # Считываем параметры и добавляем в списки
        for shape in shapes:
            if isinstance(shape, pydiffvg.Path):
                point_params.append(shape.points.requires_grad_())
                stroke_width_params.append(shape.stroke_width.requires_grad_())
        for shape_group in shape_groups:
            if isinstance(shape_group.fill_color, torch.Tensor):
                color_params.append(shape_group.fill_color.requires_grad_())
            elif isinstance(shape_group.fill_color, pydiffvg.LinearGradient):
                point_params.append(shape_group.fill_color.begin.requires_grad_())
                point_params.append(shape_group.fill_color.end.requires_grad_())
                color_params.append(shape_group.fill_color.stop_colors.requires_grad_())
            if isinstance(shape_group.stroke_color, torch.Tensor):
                color_params.append(shape_group.stroke_color.requires_grad_())
            elif isinstance(shape_group.stroke_color, pydiffvg.LinearGradient):
                point_params.append(shape_group.stroke_color.begin.requires_grad_())
                point_params.append(shape_group.stroke_color.end.requires_grad_())
                color_params.append(shape_group.stroke_color.stop_colors.requires_grad_())

        # Оптимайзеры
        point_optimizer = optim.Adam(point_params, lr=point_rate)
        color_optimizer = optim.Adam(color_params, lr=color_rate)
        stroke_width_optimizers = optim.Adam(stroke_width_params, lr=width_rate)

        # Главный цикл
        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:
            print('iteration:', run[0])
            # Зануляем градиенты
            point_optimizer.zero_grad()
            color_optimizer.zero_grad()
            stroke_width_optimizers.zero_grad()

            # Растеризуем исходную картинку по параметрам объектов
            scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height,
                                                                 shapes, shape_groups)
            render = pydiffvg.RenderFunction.apply
            img = render(canvas_width,
                         canvas_height,
                         2,
                         2,
                         0,
                         None,
                         *scene_args)

            # получаем контур текущего изображения
            contour = get_contours(canvas_width, canvas_height, shapes, shape_groups).detach()

            # Сохраняем промежуточный результат
            pydiffvg.imwrite(img.cpu(),
                             'results/style_transfer/step_{}.png'.format(run[0]),
                             gamma=1.0)

            # Сохраняем промежуточный результат в формате .svg
            pydiffvg.save_svg('results/style_transfer/step_{}.svg'.format(run[0]), canvas_width, canvas_height, shapes,
                              shape_groups)

            img = img[:, :, :3].unsqueeze(0).permute(0, 3, 1, 2).to(device)

            # Подаем модели картинки и считаем loss
            loss = perception_loss(img, style_img) + 100 * line_loss(contour)

            print('render loss:', loss.item())
            print()

            loss.backward()

            # делаем шаг оптимайзеров
            point_optimizer.step()
            color_optimizer.step()
            stroke_width_optimizers.step()

            for color in color_params:
                color.data.clamp_(0.0, 1.0)
            for w in stroke_width_params:
                w.data.clamp_(0.0, 1.0)

            # Делаем шаг
            run[0] += 1

        return shapes, shape_groups

    tic = time.perf_counter()

    # нормализуем изображения стиля и контента

    # Получаем обновленные объекты
    shapes, shape_groups = run_style_transfer(content_img, style_img,
                                              canvas_width, canvas_height,
                                              shapes, shape_groups,
                                              num_iter, point_rate,
                                              color_rate, width_rate)

    # Сохраняем итоговое векторное изображение
    with tempfile.NamedTemporaryFile() as f:
        pydiffvg.save_svg(f.name, canvas_width, canvas_height, shapes, shape_groups)
        result = open(f.name).read()

    toc = time.perf_counter()
    print(f"Time: {toc - tic:0.4f} seconds")

    return result


def diffvg_render_svg(canvas_width, canvas_height, shapes, shape_groups):
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply
    return render(canvas_width,
                  canvas_height,
                  2,
                  2,
                  0,
                  None,
                  *scene_args)


def run_vector_style_transfer(style_file, content_file):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        style_png_file = f.name
        x = canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(style_file)
        style_img = diffvg_render_svg(*x)
        print("style_png_file:", style_png_file)
        # os.makedirs(style_png_file, exist_ok=True)
        pydiffvg.imwrite(style_img.cpu(), style_png_file, gamma=1.0)

    y = canvas_width_, canvas_height_, shapes_, shape_groups_ = pydiffvg.svg_to_scene(content_file)
    content_img = diffvg_render_svg(*y)

    # num_iters
    if abs(len(shapes) - len(shapes_)) <= 30:
        num_iters = 60
    else:
        maxs = max(len(shapes_), len(shapes))
        mins = min(len(shapes_), len(shapes))
        num_iters = min(60 + 2 * maxs // mins + (canvas_width_ + canvas_height_) // 100, 200)

    # rates
    if len(shapes_) < 300:
        point_rate = 0.2
        color_rate = 0.005
        width_rate = 0.001
    elif len(shapes_) < 1000:
        point_rate = 0.3
        color_rate = 0.004
        width_rate = 0.1
    elif len(shapes_) < 1600:
        point_rate = 0.4
        color_rate = 0.01
        width_rate = 0.1
    else:
        point_rate = 0.8
        color_rate = 0.001
        width_rate = 0.1

    print(len(shapes_))
    print(point_rate)
    print(color_rate)
    print(width_rate)
    print(num_iters)

    res = transfer_style_vector_image(content_img, style_png_file,
                                      canvas_width_, canvas_height_,
                                      shapes_, shape_groups_,
                                      num_iters, point_rate,
                                      color_rate, width_rate)
    os.remove(style_png_file)
    return res
