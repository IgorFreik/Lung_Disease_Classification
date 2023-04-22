import urllib.request
from pathlib import Path
import os
import tarfile
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm


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
            fn = './data/images_%02d.tar.gz' % (idx+1)
            print('Downloading ' + fn)
            urllib.request.urlretrieve(link, fn)  # download the zip file

        print("Compressed files downloaded. Starting extraction!")

        # Uncompress data.
        for idx in range(len(links)):
            print(f'Extracting: {idx+1} / {len(links)}')

            file = tarfile.open('./data/images_%02d.tar.gz' % (idx+1))
            file.extractall('./data/')
            file.close()
            os.remove('./data/images_%02d.tar.gz' % (idx+1))

        labels = pd.read_csv('../data/Data_Entry_2017_v2020.csv')
        one_word_labels = labels['Finding Labels'].apply(lambda string: string.split('|')[0])
        lab_enc = LabelEncoder()
        one_word_labels = lab_enc.fit_transform(one_word_labels)
        np.save('./data/labels.npy', one_word_labels)

        for idx, img_name in tqdm(enumerate(os.listdir('data/images'))):
            src = os.path.join('./data/images', img_name)
            destination = os.path.join(f'./data/images_{idx//1000}', img_name)
            if not os.path.exists(f'./data/images_{idx//1000}'):
                os.mkdir(f'./data/images_{idx//1000}')
            os.replace(src, destination)
