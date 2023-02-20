import csv
from bs4 import *
import requests
import os
from PIL import Image
import numpy as np
import shutil

url_dict = {
    'wine': ['https://www.bing.com/images/search?q=wine+bottle+pictures&form=HDRSC3&first=1', 
    'https://top100.winespectator.com/', 
    'https://www.gettyimages.com/photos/wine-bottle',
    'https://www.shutterstock.com/search/wine-bottle']
}

class ExtractImages:
    """CODE MODIFIED FROM GEEKSFORGEEKS"""
    def __init__(self, urls, path):
        self.urls = urls #dict of the form {'label': [url, url]}
        self.path = path
        self.count = 0

    def download_images(self, images, label):
        #remove duplicates
        images = list(set(images))

        print(f"Total {len(images)} Image Found!")

        if len(images) != 0:
            for i, image in enumerate(images):
                # From image tag ,Fetch image Source URL

                            # 1.data-srcset
                            # 2.data-src
                            # 3.data-fallback-src
                            # 4.src
                try:

                    image_link = image["data-srcset"]
                except:
                    try:
                        image_link = image["data-src"]
                    except:
                        try:
                            # In image tag ,searching for "data-fallback-src"
                            image_link = image["data-fallback-src"]
                        except:
                            try:
                                # In image tag ,searching for "src"
                                image_link = image["src"]
                            # if no Source URL found
                            except:
                                continue
                # After getting Image Source URL
                # We will try to get the content of image
                try:
                    r = requests.get(image_link, stream=True)#.content
                    r.raw.decode_content = True

                    with open(f"{self.path}/{label}_{self.count}.png",'wb') as f:
                        shutil.copyfileobj(r.raw, f)

                    self.count += 1
                except:
                    pass

            # if all images download
            if self.count == len(images):
                print("All Images Downloaded!")
                
            # if all images not download
            else:
                print(f"Total {self.count} Images Downloaded Out of {len(images)}")

    def get_imgs(self):
        for label, urls in self.urls.items():
            for url in urls: 
                print(url)
                # content of URL
                r = requests.get(url)

                # Parse HTML Code
                soup = BeautifulSoup(r.text, 'html.parser')

                # find all images in URL
                images = soup.findAll('img')
                self.download_images(images, label)

class Numpyify:
    """Converts images to arrays: may need to csv them as well"""
    def __init__(self, folder, csv_name):
        #needs to be ran in the parent directory of image folder
        #probably need to change to be more flexible
        self.folder_path = os.getcwd() + '\\' + folder
        self.code_path = os.getcwd()
        self.csv_name = csv_name
        self.headers = ['image', 'label']

    def get_files(self):
        self._pics = []
        objects = os.listdir(self.folder_path)

        for obj in objects:
            if os.self.folder_path.isfile(obj):
                self._pics.append(obj)
            else:
                continue

        print(self.folder_path)

    def write_csv(self, mode):
        #mode is 'w', 'a', etc
        self.get_files()
        csv_ext = self.csv_name.split('.')[1]

        if csv_ext != 'csv':
            #if name doesn't have .csv extension
            self.csv_name = self.csv_name + '.csv'
        
        with open(self.csv_name, mode) as f:
            #use dictwriter: more flexible with labeling
            writer = csv.writer(f)
            writer.writerow(self.headers)

            for file in self._pics:
                label = file.split('_')[0]
                image = Image.open(file)
                image = np.asarray(image)

                if image.shape[2] == 3:
                    #convert to 2d array if 3d
                    image = image.reshape(image.shape[0], -1)

                ls = [image, label]
                writer.writerow(ls)


"""
CSV format
-----------------------
  Image  | Label      |
---------|------------|
np.array | "Wine"     |
np.array | "Not wine" |
-----------------------
"""

"""ext = ExtractImages(url_dict)
ext.get_imgs()"""
num = Numpyify('image_dataset', 'wine.csv')
num.write_csv('w')