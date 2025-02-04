import os
import argparse
import torch
from torch.utils.data import DataLoader

# Import din proiect
from ckn.data import create_svhn_dataset
from ckn.utils import accuracy
from ckn.models import UnsupCKNetSVHN

def load_args():
    parser = argparse.ArgumentParser(description="Unsupervised CKN for SVHN classification")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--datapath', type=str, default='../data/svhn/',
                        help='path to the SVHN dataset folder')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    # Straturi, subsampling, kernel-size, sigma
    parser.add_argument('--filters', nargs='+', type=int, default=[64, 128, 256],
                        help='number of filters in each layer')
    parser.add_argument('--subsamplings', nargs='+', type=int, default=[2, 2, 2],
                        help='subsampling factors in each layer')
    parser.add_argument('--kernel-sizes', nargs='+', type=int, default=[3, 3, 3],
                        help='kernel sizes for each layer')
    parser.add_argument('--sigma', nargs='+', type=float, default=[0.5, 0.5, 0.5],
                        help='parameters for dot-product kernel (e.g. 0.6, etc.)')
    parser.add_argument('--sampling-patches', type=int, default=500000,
                        help='number of subsampled patches for K-means')
    parser.add_argument('--cv', action='store_true',
        help='if True, perform cross-validation on the train set, else evaluate on test')
    # Adăugăm un argument pentru data augmentation
    parser.add_argument('--data-augmentation', action='store_true',
                        help='Use random crop and random horizontal flip on train set')

    args = parser.parse_args()
    args.gpu = torch.cuda.is_available()

    # Dacă nu vine sigma de la linia de comandă, le setăm la [0.5, 0.5, ...] etc.
    nlayers = len(args.filters)
    if len(args.sigma) < nlayers:
        default_sigma = [0.6] * nlayers
        for i in range(nlayers):
            if i >= len(args.sigma):
                args.sigma.append(default_sigma[i])

    return args

def main():
    args = load_args()
    print(args)
    torch.manual_seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)

    # 1. Creăm dataset cu data augmentation la train dacă s-a cerut
    train_dset = create_svhn_dataset(
        root=args.datapath,
        split='train',
        data_augmentation=args.data_augmentation
    )
    # 2. DataLoader-ul de train
    loader_args = {'pin_memory': True} if args.gpu else {}
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, **loader_args)

    # 3. Creăm modelul UnsupCKNetSVHN cu setările date
    model = UnsupCKNetSVHN(
        filters=args.filters,
        kernel_sizes=args.kernel_sizes,
        subsamplings=args.subsamplings,
        sigma=args.sigma
    )
    print(model)

    # 4. DataLoader-ul de test (fără augmentare)
    test_dset = create_svhn_dataset(
        root=args.datapath,
        split='test',
        data_augmentation=False
    )
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, **loader_args)

    # 5. Antrenare + validare (cross_val) sau direct test
    if args.cv:
        # face unsupervised training pentru straturile CKN
        # apoi cross_val pentru clasificator pe TOT setul de antrenament,
        # la final prezice pe test
        model.unsup_cross_val(
            data_loader=train_loader,
            test_loader=None,     # cross_val foloseste K-fold din scikit-learn
            n_sampling_patches=args.sampling_patches,
            use_cuda=args.gpu
        )
        # antrenăm clasificatorul final cu cea mai bună lambda, apoi scor pe test
        y_pred, y_true = model.predict(test_loader, use_cuda=args.gpu)
        score = accuracy(y_pred, y_true, (1,))
    else:
        # face unsupervised training pentru straturile CKN
        # și cross_val direct pe test_loader (nu e CV "real", e mai mult direct pe test)
        score = model.unsup_cross_val(
            data_loader=train_loader,
            test_loader=test_loader,
            n_sampling_patches=args.sampling_patches,
            use_cuda=args.gpu
        )

    print("Final score:", score)

if __name__ == '__main__':
    main()
