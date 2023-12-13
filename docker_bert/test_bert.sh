for i in `seq 1 10`;
        do
                python BERT_Manifest_TF.py --master_list=master_list/master_list$i --path=/manifest/manifest/ 265000 0.5 0.2 20 --comment=classic_$i
                python BERT_Manifest_TF.py --master_list=master_list/master_list$i --path=/manifest/denylist1/ 265000 0.5 0.2 20 --comment=denylist1_$i
                python BERT_Manifest_TF.py --master_list=master_list/master_list$i --path=/manifest/denylist2/ 265000 0.5 0.2 20 --comment=denylist2_$i
                python BERT_Manifest_TF.py --master_list=master_list/master_list$i --path=/manifest/permission/ 265000 0.5 0.2 20 --comment=permission_$i
                python BERT_Manifest_TF.py --master_list=master_list/master_list$i --path=/manifest/nopermission/ 265000 0.5 0.2 20 --comment=nopermission_$i
        done