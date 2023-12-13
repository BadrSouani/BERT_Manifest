python manifest_preprocess.py /manifest/denylist1/ 265000 --master_list=master_list --path=/manifest/manifest/ --taboo_list=taboo_list &
python manifest_preprocess.py /manifest/denylist2/ 265000 --master_list=master_list --path=/manifest/manifest/ --taboo_list=taboo_list_v2 &
python manifest_preprocess.py /manifest/permission/ 265000 --master_list=master_list --path=/manifest/manifest/ --xml_tag=xml_tag &
python manifest_preprocess.py /manifest/nopermission/ 265000 --master_list=master_list --path=/manifest/manifest/ --ablation=1 --xml_tag=xml_tag &
