'''cp ../text-to-image/samples/bad_gmm_samples/*.jpg images/'''

f = open('test.tsv', 'wb')

base = 10000
carry = 0 * 1000

for i in range(64):
    id = base + carry + i 
    f.write("%d\timage_%d.jpg\ttest\n" % (id, id))

f.close()

import os 
os.system("rm -rf test.tsv.tf")
os.system("python compile_data.py test.tsv myexperiment/vocab images/")
os.system("rm myexperiment/evals/*")
os.system("CUDA_VISIBLE_DEIVCES=$gpu python generate_captions.py test.tsv.tf/ myexperiment/vocab myexperiment/models/ myexperiment/evals/")
