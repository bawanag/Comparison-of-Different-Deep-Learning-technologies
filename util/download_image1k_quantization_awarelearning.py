import urllib
url, filename = ("https://s3.amazonaws.com/pytorch-tutorial-assets/imagenet_1k.zip", "imagenet_1k_data.zip")
urllib.URLopener().retrieve(url, filename)
# try: urllib.URLopener().retrieve(url, filename)
# except: urllib.request.urlretrieve(url, filename)

import zipfile
with zipfile.ZipFile("imagenet_1k_data.zip", 'r') as zip_ref:
    zip_ref.extractall("")