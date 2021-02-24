# Experiments

## exec_mnist
The first set of experiments in [QuantumFlow](https://128.84.21.199/pdf/2006.14815.pdf), as shown in Figure 2.

In exec_mnist.ipynb, we demonstrate the execution of QF-Net w/ BN on {3,6} subset of MNIST.

For other results, use the following commands for the execution.
 
Taking {3,6} subset of MNIST as an example, we list binMLP(C) w/o BN, FFNN w/o BN, MLP(C) w/o BN, QF-Net w/o BN, binMLP(C) w/ BN, FFNN w/ BN, MLP(C) w/ BN, QF-Net w/ BN, respectively.  

```console
test@linux:~$ CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -bin -nq -c "3, 6" -s 4 -e 30 -m "10, 20" -chk > log/binMLP_36_wo.res
test@linux:~$ CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -bin -c "3, 6" -s 4 -e 30 -m "10, 20" -chk > log/FFNN_36_wo.res
test@linux:~$ CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -nq -c "3, 6" -s 4 -e 30 -m "10, 20" -chk > log/MLP_36_wo.res
test@linux:~$ CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -c "3, 6" -s 4 -e 30 -m "10, 20" -chk > log/QFNET_36_wo.res
test@linux:~$ CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -wn -bin -nq -c "3, 6" -s 4 -e 30 -m "10, 20" -chk > log/binMLP_36_wo.res
test@linux:~$ CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -wn -bin -c "3, 6" -s 4 -e 30 -m "10, 20" -chk > log/FFNN_36_wo.res
test@linux:~$ CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -wn -nq -c "3, 6" -s 4 -e 30 -m "10, 20" -chk > log/MLP_36_wo.res
test@linux:~$ CUDA_VISIBLE_DEVICES=0 python exe_mnist.py -wn -c "3, 6" -s 4 -e 30 -m "10, 20" -chk > log/QFNET_36_wo.res  
```

For datasets with more than 2 classes. Please use the following script to run the experiments.

```
$dataset is the interested classes

# 3 classes
python -u exe_mnist.py -qa "-1 -1 1 1 1 -1 1 -1, -1 -1 -1" -nn "8, 3" -bin -qt -c $dataset -s 4 -l 0.1 -ql 0.0001 -e 5 -m "2, 4" 

# 4 classes
python -u exe_mnist.py -qa "1 -1 1 -1 -1 1 -1 -1 1 1 -1 -1 -1 1 1 1, -1 -1 -1 -1" -nn "16, 4" -bin -qt -c $dataset -s 8 -l 0.1 -ql 0.0001 -e 5 -m "2, 4" 

# 5 classes
python -u exe_mnist.py -qa "1 -1 1 -1 -1 1 -1 -1 1 1 -1 -1 -1 1 1 1, -1 -1 -1 -1 -1" -nn "16, 5" -bin -qt -c $dataset -s 8 -l 0.1 -ql 0.0001 -e 5 -m "2, 4"
```
