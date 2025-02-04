import os
import argparse
import torch
from torch.utils.data import DataLoader

from ckn.data import create_stl10_dataset
from ckn.utils import accuracy
from ckn.models import UnsupCKNetSTL10

def load_args():
    parser = argparse.ArgumentParser(description="Unsupervised CKN for STL-10 classification")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--datapath', type=str, default='../data/stl10/',
                        help='path to the STL10 dataset folder')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size')
    # poți modifica default-urile la nevoile tale
    parser.add_argument('--filters', nargs='+', type=int, default=[64, 128, 256])
    parser.add_argument('--subsamplings', nargs='+', type=int, default=[2, 2, 2])
    parser.add_argument('--kernel-sizes', nargs='+', type=int, default=[3, 3, 3])
    parser.add_argument('--sigma', nargs='+', type=float, default=[0.5, 0.5, 0.5])
    parser.add_argument('--sampling-patches', type=int, default=500000,
                        help='number of subsampled patches for K-means')
    parser.add_argument('--cv', action='store_true',
                        help='if True, do cross-validation on train set, else test on test')
    parser.add_argument('--data-augmentation', action='store_true',
                        help='Use random crop and random horizontal flip for train set')

    args = parser.parse_args()
    args.gpu = torch.cuda.is_available()
    return args

def main():
    args = load_args()
    print(args)

    torch.manual_seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)

    # 1. Creăm datasetul de antrenare (split='train')
    train_dset = create_stl10_dataset(
        root=args.datapath,
        split='train',
        data_augmentation=args.data_augmentation
    )

    # 2. DataLoader de train
    loader_args = {'pin_memory': True} if args.gpu else {}
    train_loader = DataLoader(train_dset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, **loader_args)

    # 3. Construim modelul UnsupCKNetSTL10
    model = UnsupCKNetSTL10(
        filters=args.filters,
        kernel_sizes=args.kernel_sizes,
        subsamplings=args.subsamplings,
        sigma=args.sigma
    )
    print(model)

    # 4. DataLoader de test (split='test')
    test_dset = create_stl10_dataset(
        root=args.datapath,
        split='test',
        data_augmentation=False
    )
    test_loader = DataLoader(test_dset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, **loader_args)

    # 5. Antrenare + validare/cross-val
    if args.cv:
        # cross-val pe train (nu folosim test_loader pe parcurs),
        # la final prezicem pe test
        model.unsup_cross_val(
            data_loader=train_loader,
            test_loader=None,
            n_sampling_patches=args.sampling_patches,
            use_cuda=args.gpu
        )
        y_pred, y_true = model.predict(test_loader, use_cuda=args.gpu)
        score = accuracy(y_pred, y_true, (1,))
    else:
        # antrenare CKN + clasificator, validare direct pe test
        score = model.unsup_cross_val(
            data_loader=train_loader,
            test_loader=test_loader,
            n_sampling_patches=args.sampling_patches,
            use_cuda=args.gpu
        )

    print("Final score:", score)

if __name__ == '__main__':
    main()
