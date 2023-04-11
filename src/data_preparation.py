import urllib.request
from pathlib import Path
import os
import tarfile
import torchvision.transforms as tt
import torch
from torch.utils.data import DataLoader
import torchvision

# URLs for the zip files. The data is publicly available at https://nihcc.app.box.com/v/ChestXray-NIHCC/.
links = [
    'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
    'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
    'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
    'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
    'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
    'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
    'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
    'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
    'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
    'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
    'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
    'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
]


def download_data():
    # Download data if not already downloaded.
    if not Path(os.path.join(os.path.dirname(__file__), '../data')).exists():
        print('Creating data folder.')
        os.mkdir(Path(os.path.join(os.path.dirname(__file__), '../data')))

        for idx, link in enumerate(links):
            fn = '../data/images_%02d.tar.gz' % (idx+1)
            print('Downloading ' + fn)
            urllib.request.urlretrieve(link, fn)  # download the zip file

        print("Download complete. Please check the checksums.")

        # Uncompress data.
        for idx in range(len(links)):
            print(f'Extracting: {idx+1} / {len(links)}')

            file = tarfile.open('../data/images_%02d.tar.gz' % (idx+1))
            file.extractall('../data/')
            file.close()


def get_loaders(train_batch_size=64, test_batch_size=32):
    data_path = '../data/images'

    full_dataset = torchvision.datasets.Imagefolder(data_path)
    train_size = int(0.8 * len(full_dataset))

    train_data = torch.utils.data.Subset(full_dataset, range(train_size))
    test_data = torch.utils.data.Subset(full_dataset, range(train_size, len(full_dataset)))

    train_data.dataset.transform = tt.Compose([
        tt.Resize((224, 224)),
        tt.RandomHorizontalFlip(p=0.5),
        tt.RandomRotation(degrees=10),
        tt.ToTensor(),
        tt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    test_data.dataset.transform = tt.Compose([
        tt.Resize((224, 224)),
        tt.ToTensor(),
        tt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader
