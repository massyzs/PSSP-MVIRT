# PSSP-MVIRT
peptide secondary structure prediction

Protein.py should be trained on protein dataset, and then train the peptide.py on the saved model generated by Protein.py.
The dataset is available on our server:http://server.malab.cn/PSSP-MVIRT (if not avalible, please contact corresponding author professor Wei)
(Email of first author Cao: caoxiao.sdu@bytedance.com||1265199717@qq.com)


For the whole project (used for testset),please download on https://pan.baidu.com/s/1nCnD0LRnyKDM7IIFKqNcJQ  Password: m90q (suggest)
Usage
Step1:
unzip the model file and dataset (7z format)
Step2:
you can simply run by: 
python peptide-1.25.py --test_num -1 --model ./model/complex-1.25-t-v4.model --data ./cxdatabase/withoutX --output_dir ./
Step3:
check the output sequence in --output_dir+'./predicted_seq.txt'
check the performance in --output_dir+'./evaluate.txt'

