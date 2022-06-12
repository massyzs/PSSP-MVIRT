# PSSP-MVIRT (peptide secondary structure prediction)

Server:http://server.malab.cn/PSSP-MVIRT 

## Download
For the project (used for testset),please download on [Baidu Drive](https://pan.baidu.com/s/1nCnD0LRnyKDM7IIFKqNcJQ) Password: m90q 
## Usage
### Step1
unzip the model file and dataset (7z format)
### Step2
you can simply run by:

```bash
python peptide-1.25.py --test_num -1 --model ./model/complex-1.25-t-v4.model --data ./cxdatabase/withoutX --output_dir ./
```

### Step3
check the output sequence in --output_dir+'./predicted_seq.txt'
check the performance in --output_dir+'./evaluate.txt'

## Email
Email of the first author Cao: caoxiao.sdu@bytedance.com||1265199717@qq.com(prefer)||shaocao@mail.sdu.edu.cn

## Notification
Only for academic use. Commercial use is forbidden.
