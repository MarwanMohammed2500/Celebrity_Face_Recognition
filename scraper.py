import os
from icrawler.builtin import BingImageCrawler

def download_celeb_images(celeb_names, max_num=10, output_dir="celeb_images"):
    os.makedirs(output_dir, exist_ok=True)

    for name in celeb_names:
        print(f"Downloading images for: {name}")
        name_folder = os.path.join(output_dir, name.replace(" ", "_"))
        os.makedirs(name_folder, exist_ok=True)

        crawler = BingImageCrawler(storage={'root_dir': name_folder})
        crawler.crawl(keyword=name + " face", max_num=max_num)