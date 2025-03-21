# SpatialLM

<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="figures/logo_light.png#gh-light-mode-only" width="60%" alt="SpatialLM" />
  <img src="figures/logo_dark.png#gh-dark-mode-only" width="60%" alt="SpatialLM" />
</div>
<hr style="margin-top: 0; margin-bottom: 8px;">
<div align="center" style="margin-top: 0; padding-top: 0; line-height: 1;">
    <a href="https://manycore-research.github.io/SpatialLM" target="_blank" style="margin: 2px;"><img alt="Project"
    src="https://img.shields.io/badge/🌐%20Website-SpatialLM-ffc107?color=42a5f5&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
    <a href="https://github.com/manycore-research/SpatialLM" target="_blank" style="margin: 2px;"><img alt="GitHub"
    src="https://img.shields.io/badge/GitHub-SpatialLM-24292e?logo=github&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
</div>
<div align="center" style="line-height: 1;">
    <a href="https://huggingface.co/manycore-research/SpatialLM-Llama-1B" target="_blank" style="margin: 2px;"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SpatialLM%201B-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
    <a href="https://huggingface.co/datasets/manycore-research/SpatialLM-Testset" target="_blank" style="margin: 2px;"><img alt="Dataset"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-SpatialLM-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
</div>

## Introduction

SpatialLM is a 3D large language model designed to process 3D point cloud data and generate structured 3D scene understanding outputs. These outputs include architectural elements like walls, doors, windows, and oriented object bounding boxes with their semantic categories. Unlike previous methods that require specialized equipment for data collection, SpatialLM can handle point clouds from diverse sources such as monocular video sequences, RGBD images, and LiDAR sensors. This multimodal architecture effectively bridges the gap between unstructured 3D geometric data and structured 3D representations, offering high-level semantic understanding. It enhances spatial reasoning capabilities for applications in embodied robotics, autonomous navigation, and other complex 3D scene analysis tasks.

<div align="center">
  <video src="https://github.com/user-attachments/assets/c0218d6a-f676-41f8-ae76-bba228866306" poster="figures/cover.png"> </video>
  <p><i>SpatialLM reconstructs 3D layout from a monocular RGB video with MASt3R-SLAM. Results aligned to video with GT cameras for visualization.</i></p>
</div>

## SpatialLM Models

<div align="center">

|      **Model**      | **Download**                                                                   |
| :-----------------: | ------------------------------------------------------------------------------ |
| SpatialLM-Llama-1B  | [🤗 HuggingFace](https://huggingface.co/manycore-research/SpatialLM-Llama-1B)  |
| SpatialLM-Qwen-0.5B | [🤗 HuggingFace](https://huggingface.co/manycore-research/SpatialLM-Qwen-0.5B) |

</div>

## Usage

### Installation

Tested with the following environment:

- Python 3.11
- Pytorch 2.4.1
- CUDA Version 12.4

```bash
# clone the repository
git clone https://github.com/manycore-research/SpatialLM.git
cd SpatialLM

# create a conda environment with cuda 12.4
conda create -n spatiallm python=3.11
conda activate spatiallm
conda install -y nvidia/label/cuda-12.4.0::cuda-toolkit conda-forge::sparsehash

# Install dependencies with poetry
pip install poetry && poetry config virtualenvs.create false --local
poetry install
poe install-torchsparse # Building wheel for torchsparse will take a while
```

### Inference

In the current version of SpatialLM, input point clouds are considered axis-aligned where the z-axis is the up axis. This orientation is crucial for maintaining consistency in spatial understanding and scene interpretation across different datasets and applications.
Example preprocessed point clouds, reconstructed from RGB videos using [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM), are available in [SpatialLM-Testset](#spatiallm-testset).

Download an example point cloud:

```bash
huggingface-cli download manycore-research/SpatialLM-Testset pcd/scene0000_00.ply --repo-type dataset --local-dir .
```

Run inference:

```bash
python inference.py --point_cloud pcd/scene0000_00.ply --output scene0000_00.txt --model_path manycore-research/SpatialLM-Llama-1B
```

### Visualization

Use `rerun` to visualize the point cloud and the predicted structured 3D layout output:

```bash
# Convert the predicted layout to Rerun format
python visualize.py --point_cloud pcd/scene0000_00.ply --layout scene0000_00.txt --save scene0000_00.rrd

# Visualize the point cloud and the predicted layout
rerun scene0000_00.rrd
```

### Evaluation

To evaluate the performance of SpatialLM, we provide an `eval.py` script that reports the benchmark results on the SpatialLM-Testset in the table below in section [Benchmark Results](#benchmark-results).

Download the testset:

```bash
huggingface-cli download manycore-research/SpatialLM-Testset --repo-type dataset --local-dir SpatialLM-Testset
```

Run evaluation:

```bash
# Run inference on the PLY point clouds in folder SpatialLM-Testset/pcd with SpatialLM-Llama-1B model
python inference.py --point_cloud SpatialLM-Testset/pcd --output SpatialLM-Testset/pred --model_path manycore-research/SpatialLM-Llama-1B

# Evaluate the predicted layouts
python eval.py --metadata SpatialLM-Testset/test.csv --gt_dir SpatialLM-Testset/layout --pred_dir SpatialLM-Testset/pred --label_mapping SpatialLM-Testset/benchmark_categories.tsv
```

## SpatialLM Testset

We provide a test set of 107 preprocessed point clouds, reconstructed from RGB videos using [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM). SpatialLM-Testset is quite challenging compared to prior clean RGBD scans datasets due to the noise and occlusions in the point clouds reconstructed from monocular RGB videos.

<div align="center">

|    **Dataset**    | **Download**                                                                       |
| :---------------: | ---------------------------------------------------------------------------------- |
| SpatialLM-Testset | [🤗 Datasets](https://huggingface.co/datasets/manycore-research/SpatialLM-TestSet) |

</div>

## Benchmark Results

Benchmark results on the challenging SpatialLM-Testset are reported in the following table:

<div align="center">

| **Method**       | **SpatialLM-Llama-1B** | **SpatialLM-Qwen-0.5B** |
| ---------------- | ---------------------- | ----------------------- |
| **Floorplan**    | **mean IoU**           |                         |
| wall             | 78.62                  | 74.81                   |
|                  |                        |                         |
| **Objects**      | **F1 @.25 IoU (3D)**   |                         |
| curtain          | 27.35                  | 28.59                   |
| nightstand       | 57.47                  | 54.39                   |
| chandelier       | 38.92                  | 40.12                   |
| wardrobe         | 23.33                  | 30.60                   |
| bed              | 95.24                  | 93.75                   |
| sofa             | 65.50                  | 66.15                   |
| chair            | 21.26                  | 14.94                   |
| cabinet          | 8.47                   | 8.44                    |
| dining table     | 54.26                  | 56.10                   |
| plants           | 20.68                  | 26.46                   |
| tv cabinet       | 33.33                  | 10.26                   |
| coffee table     | 50.00                  | 55.56                   |
| side table       | 7.60                   | 2.17                    |
| air conditioner  | 20.00                  | 13.04                   |
| dresser          | 46.67                  | 23.53                   |
|                  |                        |                         |
| **Thin Objects** | **F1 @.25 IoU (2D)**   |                         |
| painting         | 50.04                  | 53.81                   |
| carpet           | 31.76                  | 45.31                   |
| tv               | 67.31                  | 52.29                   |
| door             | 50.35                  | 42.15                   |
| window           | 45.4                   | 45.9                    |

</div>

## License

SpatialLM-Llama-1B is derived from Llama3.2-1B-Instruct, which is licensed under the Llama3.2 license.
SpatialLM-Qwen-0.5B is derived from the Qwen-2.5 series, originally licensed under the Apache 2.0 License.

All models are built upon the SceneScript point cloud encoder, licensed under the CC-BY-NC-4.0 License. TorchSparse, utilized in this project, is licensed under the MIT License.

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{spatiallm,
  title        = {SpatialLM: Large Language Model for Spatial Understanding},
  author       = {ManyCore Research Team},
  howpublished = {\url{https://github.com/manycore-research/SpatialLM}},
  year         = {2025}
}
```

## Acknowledgements

We would like to thank the following projects that made this work possible:

[Llama3.2](https://github.com/meta-llama) | [Qwen2.5](https://github.com/QwenLM/Qwen2.5) | [Transformers](https://github.com/huggingface/transformers) | [SceneScript](https://github.com/facebookresearch/scenescript) | [TorchSparse](https://github.com/mit-han-lab/torchsparse)
