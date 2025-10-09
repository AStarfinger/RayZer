<div align="center">

# RayZer: A Self-supervised Large View Synthesis Model 

### ICCV 2025 (Oral)

<p align="center">  
    <a href="https://hwjiang1510.github.io/">Hanwen Jiang</a>,
    <a href="https://www.cs.unc.edu/~airsplay/">Hao Tan</a>,
    <a href="https://quartz-khaan-c6f.notion.site/Peng-Wang-0ab0a2521ecf40f5836581770c14219c">Peng Wang</a>,
    <a href="https://haian-jin.github.io/">Haian Jin</a>,
    <a href="https://zhaoyue-zephyrus.github.io/">Yue Zhao</a>,
    <a href="https://sai-bi.github.io/">Sai Bi</a>,
    <a href="https://kai-46.github.io/website/">Kai Zhang</a>,
    <a href="https://luanfujun.com/">Fujun Luan</a>,
    <a href="http://www.kalyans.org/">Kalyan Sunkavalli</a>,
    <a href="https://www.cs.utexas.edu/~huangqx/index.html">Qixing Huang</a>,
    <a href="https://geopavlakos.github.io/">Georgios Pavlakos</a>

</p>


</div>


<div align="center">
    <a href="https://hwjiang1510.github.io/RayZer/"><strong>Project Page</strong></a> |
    <a href="https://arxiv.org/abs/2505.00702"><strong>Paper</strong></a> 
</div>

--------------------------------------------------------------------------------
<br>


## 0. Clarification

This is the **official repository** for the paper _"RayZer: A Self-supervised Large View Synthesis Model "_.

The code here is a **re-implementation** and **differs** from the original version developed at Adobe. However, the provided checkpoints are from the original Adobe implementation and were trained inside Adobe. This codebase is developed based on <a href="https://github.com/Haian-Jin/LVSM?tab=readme-ov-file"><strong>LVSM</strong></a>.

We have verified that the re-implemented version matches the performance of the original. For any questions or issues, please contact Hanwen Jiang at [hwjiang1510@gmail.com](mailto:hwjiang1510@gmail.com).

---



## 1. Preparation

### Environment
```
conda create -n rayzer python=3.11
conda activate rayzer
pip install -r requirements.txt
```
As we used [xformers](https://github.com/facebookresearch/xformers) `memory_efficient_attention`, the GPU device compute capability needs > 8.0. Otherwise, it would pop up an error. Check your GPU compute capability in [CUDA GPUs Page](https://developer.nvidia.com/cuda-gpus#compute).


### Data
We provide preprocessed [DL3DV](https://dl3dv-10k.github.io/DL3DV-10K/) benchmark data for evaluation purpose. You can find the preprocessed data [here](https://huggingface.co/datasets/hwjiang/DL3DV-benchmark-preprocessed/resolve/main/dl3dv_benchmark.zip?download=true). Then place the data at ```./dl3dv_benchmark```

Note that for training, you will need to preprocess the training set (10K scenes not included in benchmark) as the same data format.

### Checkpoints

| Data | Model | View Sampling | PSNR  | SSIM  | LPIPS | Results |
|------|-------|---------------|-------|-------|-------|---------|
| DL3DV | [RayZer-8-12-12-100K](https://huggingface.co/hwjiang/RayZer/resolve/main/rayzer_dl3dv_8_12_12_96k.pt?download=true) | Even | 25.59 | 0.795 | 0.183 | [link](https://drive.google.com/file/d/1x4M43rSc8KKJK9IZsNioqvRtNnbGP5NF/view?usp=sharing) |
| DL3DV | [RayZer-8-12-12-100K](https://huggingface.co/hwjiang/RayZer/resolve/main/rayzer_dl3dv_8_12_12_96k.pt?download=true) | Random | 25.47 | 0.795 | 0.181 | [link](https://drive.google.com/file/d/1x4M43rSc8KKJK9IZsNioqvRtNnbGP5NF/view?usp=sharing) |


## 2. Training

Before training, you need to follow the instructions [here](https://docs.wandb.ai/guides/track/public-api-guide/#:~:text=You%20can%20generate%20an%20API,in%20the%20upper%20right%20corner.) to generate the Wandb key file for logging and save it in the `configs` folder as `api_keys.yaml`. You can use the `configs/api_keys_example.yaml` as a template.

Note that for training, you will need to preprocess the training set (10K scenes not included in benchmark) as the same data format.

The original training command:
```bash
torchrun --nproc_per_node 8 --nnodes 4 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29502 \
    train.py --config configs/rayzer_dl3dv.yaml
```
The training will be distributed across 8 GPUs and 4 nodes with a total batch size of 256.
`rayzer_dl3dv.yaml` is the config file for the RayZer-DL3DV model. You can also use `LVSM_dl3dv.yaml` for training LVSM assuming known poses.
Note that for efficiency, we use a patch size of 16, which is different from the patch size of 8 used in original LVSM paper.



## 3. Inference

```bash
torchrun --nproc_per_node 8 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
    inference.py --config "configs/rayzer_dl3dv.yaml" \
    training.dataset_path = "./data/dl3dv10k_benchmark.txt" \
    training.batch_size_per_gpu = 4 \
    training.target_has_input =  false \
    training.num_views = 24 \
    training.num_input_views = 16 \
    training.num_target_views = 8 \
    inference.if_inference = true \
    inference.compute_metrics = true \
    inference.render_video = false \
    inference.view_idx_file_path = "./data/rayzer_evaluation_index_dl3dv_even.json" \
    inference.model_path = ./model_checkpoints/rayzer_dl3dv_8_12_12_96k.pt \
    inference_out_root = ./experiments/evaluation/test \
```
We use `./data/rayzer_evaluation_index_dl3dv_even.json` and `./data/rayzer_evaluation_index_dl3dv_random.json` to specify the view indices. The two files correspond to the settings of even sampling and random sampling in paper.

After the inference, the code will generate a html file in the `inference_out_dir` folder. You can open the html file to view the results.

## 4. Citation 

If you find this work useful in your research, please consider citing:

```bibtex
@article{jiang2025rayzer,
  title={RayZer: A Self-supervised Large View Synthesis Model},
  author={Jiang, Hanwen and Tan, Hao and Wang, Peng and Jin, Haian and Zhao, Yue and Bi, Sai and Zhang, Kai and Luan, Fujun and Sunkavalli, Kalyan and Huang, Qixing and others},
  journal={arXiv preprint arXiv:2505.00702},
  year={2025}
}
```

## 5. TODO
[] Prepare evaluation and training scripts on RE10K.


## 6. Known Issues
The model can be sensitive to the number of views, as it uses image index embedding. Make sure your number of views are the same during training and testing.

