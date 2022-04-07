import numpy as np
import os
import pandas as pd
import pyro
import torch

from joblib import Parallel, delayed

from pyro.distributions import Gamma, Normal, TransformedDistribution
from pyro.distributions.transforms import SigmoidTransform, AffineTransform, ComposeTransform

from tqdm import tqdm

from deepscm.datasets.morphomnist import load_morphomnist_like, save_morphomnist_like
from deepscm.datasets.morphomnist.transforms import SetThickness, ImageMorphology

NUM_CORES = 8


def get_intensity(img):
    threshold = 0.5

    img_min, img_max = img.min(), img.max()
    mask = (img >= img_min + (img_max - img_min) * threshold)
    avg_intensity = np.median(img[mask])

    return avg_intensity


def model(n_samples=None, scale=0.5, invert=False):
    with pyro.plate('observations', n_samples):
        thickness = 0.5 + pyro.sample('thickness', Gamma(10., 5.))

        if invert:
            loc = (thickness - 2) * -2
        else:
            loc = (thickness - 2.5) * 2

        transforms = ComposeTransform([SigmoidTransform(), AffineTransform(64, 191)])

        intensity = pyro.sample('intensity', TransformedDistribution(Normal(loc, scale), transforms))

    return thickness, intensity

def get_covariate_thickness_intensity(labels, images, change_thickness=True, change_intensity=False, invert=False):
    # default digit class: [0, 9]
    # desired thickness mapping: [1, 5]
        # let's do linear mapping to [1.5, 4.5] and then let 2sd's hit the limits (2sd = 0.5)
        # remember to clip at like 0.5
    # desired intensity mapping: [50, 250]
        # let's do linear mapping to [75, 225] and then let 2sd's hit the limits (2sd = 25)
        # remember to clip at like 50 and 256

    n_samples = len(labels)
    
    if change_thickness:
        thickness = []
        for image_idx, label in enumerate(labels):
            thickness_affine_min = 1
            thickness_affine_max = 4.5
            thickness_range = thickness_affine_max - thickness_affine_min

            thickness_value = thickness_affine_min + label / 9. * (thickness_range)
            thickness_value = thickness_value + np.random.normal(0, 0.5)

            if invert: 
                thickness_value = 5.5 - thickness_value

            thickness_value = max(thickness_value, 0.5)
            thickness_value = min(thickness_value, 5.25)


            thickness.append(thickness_value)
    else: 
        thickness = []
        for image_idx, image in enumerate(images):
            morph = ImageMorphology(image, scale=16)
            thickness.append(morph.mean_thickness)

    if change_intensity: 
        intensity = []
        for image_idx, label in enumerate(labels):
            intensity_affine_min = 125 
            intensity_affine_max = 225
            intensity_range = intensity_affine_max - intensity_affine_min

            intensity_value = intensity_affine_min + label / 9. * (intensity_range)
            intensity_value = intensity_value + np.random.normal(0, 25)

            if invert: 
                intensity_value = 255 - intensity_value + 75 

            intensity_value = max(intensity_value, 75)
            intensity_value = min(intensity_value, 255)


            intensity.append(intensity_value)
    else: 
        intensity = [get_intensity(img) for img in images]
        #for image_idx, image in enumerate(images):
        #    morph = ImageMorphology(image, scale=16)
        #    if image_idx % 100 == 0: 
        #        print(image_idx)
        #    intensity.append(get_intensity(morph.image))

    return thickness, intensity

    

def gen_dataset(args, train=True):
    pyro.clear_param_store()
    images_, labels, _ = load_morphomnist_like(args.data_dir, train=train)

    if args.digit_class is not None:
        mask = (labels == args.digit_class)
        images_ = images_[mask]
        labels = labels[mask]

    images = np.zeros_like(images_)

    n_samples = len(images)
    #import pdb; pdb.set_trace()
    thickness, intensity = get_covariate_thickness_intensity(labels, images_, args.change_thickness, args.change_intensity, invert=not train)
    #import pdb; pdb.set_trace()
    #with torch.no_grad():
        #thickness, intensity = model(n_samples, scale=args.scale, invert=args.invert)

    metrics = pd.DataFrame(data={'thickness': thickness, 'intensity': intensity})

    #for n, (thickness, intensity) in enumerate(tqdm(zip(thickness, intensity), total=n_samples)):

    def get_processed_image(n, thickness, intensity):
        morph = ImageMorphology(images_[n], scale=16)

        if args.change_thickness: 
            tmp_img = morph.downscale(np.float32(SetThickness(thickness)(morph)))
        else: 
            tmp_img = morph.downscale(np.float32(morph))

        avg_intensity = get_intensity(tmp_img)

         
        mult = intensity / avg_intensity
        
        if args.change_intensity: 
            tmp_img = np.clip(tmp_img * mult, 0, 255)

        return tmp_img

    testing = False

    if testing: 
        thickness = thickness[:100]
        intensity = intensity[:100]
        labels = labels[:100]
        metrics = metrics[:100]
        n_samples = 100

    images = Parallel(n_jobs=NUM_CORES)(delayed(get_processed_image)(n, thickness, intensity) for n, (thickness, intensity) in enumerate(tqdm(zip(thickness, intensity), total=n_samples)))


    save_morphomnist_like(images, labels, metrics, args.out_dir, train=train)

"""
    for n, (thickness, intensity) in enumerate(tqdm(zip(thickness, intensity), total=n_samples)):
        #if labels[n] == 0:
            #import pdb; pdb.set_trace()

        #if labels[n] == 9: 
            #import pdb; pdb.set_trace()

        if n == 100: 
            #break
            pass

        morph = ImageMorphology(images_[n], scale=16)

        if args.change_thickness: 
            tmp_img = morph.downscale(np.float32(SetThickness(thickness)(morph)))
        else: 
            tmp_img = morph.downscale(np.float32(morph))

        avg_intensity = get_intensity(tmp_img)

         
        mult = intensity / avg_intensity
        
        if args.change_intensity: 
            tmp_img = np.clip(tmp_img * mult, 0, 255)

        images[n] = tmp_img

"""


    # TODO: do we want to save the sampled or the measured metrics?



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--change-thickness', default=True, action='store_true')
    parser.add_argument('--change-intensity', default=False, action='store_true')
    parser.add_argument('--data-dir', type=str, default='/vol/biomedic/users/dc315/mnist/original/', help="Path to MNIST (default: %(default)s)")
    parser.add_argument('-o', '--out-dir', type=str, help="Path to store new dataset")
    parser.add_argument('-d', '--digit-class', type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], help="digit class to select")
    parser.add_argument('-s', '--scale', type=float, default=0.5, help="scale of logit normal")
    parser.add_argument('-i', '--invert', default=False, action='store_true', help="inverses correlation")

    args = parser.parse_args()

    print(f'Generating data for:\n {args.__dict__}')

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'args.txt'), 'w') as f:
        print(f'Generated data for:\n {args.__dict__}', file=f)

    print('Generating Training Set')
    print('#######################')
    gen_dataset(args, True)

    print('Generating Test Set')
    print('###################')
    gen_dataset(args, False)
