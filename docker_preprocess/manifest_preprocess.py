import os
import pandas as pd
import argparse
from shutil import copy

def dir_path(path: str):
    if os.path.exists(path):
        return path
    else:
        print(f"PATH is {path}")
        raise NotADirectoryError(path)


parser = argparse.ArgumentParser(description='Process values.')
parser.add_argument('--master_list', type=str, default="master_list", help="Path of the Manifest list")
parser.add_argument('--path', type=dir_path, default="/manifest/manifest/", help="Path of the Manifest directory")
parser.add_argument("--ablation", default=0,
                    help=" By default xml_tag's tags are kept, put the flag if you want to delete them from manifests instead",
                    type=bool)
parser.add_argument("--xml_tag", default="",
                    help="Path to file with tags to keep or delete before application of the taboo_list", type=str)
parser.add_argument("--taboo_list", nargs='?', default="", help='Path to taboo list for manifests', type=str)
parser.add_argument("preprocessed_path", type=str, help="Path for the generated manifests")
parser.add_argument("N", help="Number of Application to use", type=int)

args = parser.parse_args()

preprocessed_path = args.preprocessed_path
ablation = args.ablation
MANIFESTS_DIR = args.path
TABOO_DIR = args.taboo_list
TAGS_DIR = args.xml_tag
master_list_name = args.master_list
nApp = args.N

print(f"PATH of master_list: {master_list_name}")
master_list = pd.read_csv(master_list_name, header=None).to_numpy()
truncated_array = master_list[:nApp]
manifest = ["" for i in range(nApp)]

if not os.path.exists(preprocessed_path):
    os.makedirs(preprocessed_path)

# No tag selection
if "" == TAGS_DIR:
    for i in range(nApp):
        f_in = open(MANIFESTS_DIR + truncated_array[i][0], encoding='utf-8')
        manifest[i] = f_in.read()
        f_in.close()

# Tag selection
else:
    copy(TAGS_DIR, preprocessed_path)
    manifest_content = {}
    tag_names = open(TAGS_DIR, 'r').readlines()
    len_tag = len(tag_names)
    for i in range(len_tag):
        tag_names[i] = tag_names[i].replace('\n', '')
    # Per manifest
    for i in range(nApp):
        f_in = open(MANIFESTS_DIR + truncated_array[i][0], encoding='utf-8')
        manifest_content[i] = f_in.readlines()
        f_in.close()
        manifest_filtered = ''
        # per line
        for line in manifest_content[i]:
            keep = ablation
            counter = 0
            while ((keep == 0 and ablation == 0) or (keep and ablation)) and counter < len_tag:
                if tag_names[counter] in line : keep = 1-ablation
                counter += 1
            if keep : manifest_filtered += line
        manifest[i] = manifest_filtered

# Taboo list usage
if not "" == TABOO_DIR:
    copy(TABOO_DIR, preprocessed_path)
    taboo = open(TABOO_DIR, 'r').readlines()
    for i in range(len(taboo)):
        taboo[i] = taboo[i].replace('\n', '')
    for i in range(nApp):
        for word in taboo:
            manifest[i] = manifest[i].replace(word, '')



for i in range(nApp):
    text_file = open(preprocessed_path+truncated_array[i][0], "w", encoding="utf-8")
    text_file.write(manifest[i])
    text_file.close()



