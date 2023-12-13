import requests
from androguard.core.bytecodes.apk import APK
import os
import sys

APIKEY = ''
if (APIKEY == None) or (len(APIKEY) != 64):
    print("APIKEY not set. Aborting")
    sys.exit(1)
MANIFEST_DIR = '/manifest/manifest'
if not os.path.isdir(MANIFEST_DIR):
    os.makedirs(MANIFEST_DIR, exist_ok=True)


def get_apk_from_androzoo(sha256):
    r = requests.get('https://androzoo.uni.lu/api/download', params={'sha256': sha256, 'apikey': APIKEY})
    if r.status_code == 200:
        return r.content
    return None

def get_manifest_from_apk_blob(apk_blob):
    a = APK(apk_blob, raw=True)
    return a.get_android_manifest_axml().get_xml()


def get_save_manifest_from_sha(sha256):
    apk_blob = get_apk_from_androzoo(sha256)
    manifest = get_manifest_from_apk_blob(apk_blob)
    with open(os.path.join(MANIFEST_DIR, sha256), 'wb') as fout:
        fout.write(manifest)
    return


if __name__ == "__main__":
    # If arguuments: process those
    # otherwise read from stdin
    if len(sys.argv) > 2:
        for sha in sys.argv[1:]:
            get_save_manifest_from_sha(sha)
    else:
        print(">>>Opening sha file\n")
        file = open(sys.argv[1], mode="r")
        for sha in file.readlines():
            sha = sha.strip().partition(",")[0]
            print(">>>>>Getting manifest of :"+sha+"\n")
            get_save_manifest_from_sha(sha)
