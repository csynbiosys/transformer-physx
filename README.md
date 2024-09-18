# Transformer PhysX

Transformer PhysX is a Python packaged modeled after the [Hugging Face repository](https://github.com/huggingface/transformers) designed for the use of transformers for modeling physical systems. Transformers have seen recent success in both natural language processing and vision fields but have yet to fully permute other machine learning areas. Originally proposed in [Transformers for Modeling Physical Systems](https://arxiv.org/abs/2010.03957), this projects goal is to make these deep learning advances including self-attention and Koopman embeddings more accessible for the scientific machine learning community.

[Documentation](https://transformer-physx.readthedocs.io) | [Getting Started](https://transformer-physx.readthedocs.io/en/latest/install.html) | [Data](https://www.doi.org/10.5281/zenodo.5148523)

### Associated Papers

Transformers for Modeling Physical Systems [ [ArXiV](https://arxiv.org/abs/2010.03957) ] [ [Neural Networks](https://www.sciencedirect.com/science/article/abs/pii/S0893608021004500) ]


### Additional Resources

- [Huggingface Repository](https://github.com/huggingface/transformers)
- [Transformer illustrated blog post](https://jalammar.github.io/illustrated-transformer/)
- [Deep learning Koopman dynamics blog post](https://nicholasgeneva.com/deep-learning/koopman/dynamics/2020/05/30/intro-to-koopman.html)


### Contact
Open an issue on the Github repository if you have any questions/concerns.

### Citation
Find this useful or like this work? Cite us with:

```latex
@article{geneva2022transformers,
    title = {Transformers for modeling physical systems},
    author = {Nicholas Geneva and Nicholas Zabaras},
    journal = {Neural Networks},
    volume = {146},
    pages = {272-289},
    year = {2022},
    issn = {0893-6080},
    doi = {10.1016/j.neunet.2021.11.022},
    url = {https://www.sciencedirect.com/science/article/pii/S0893608021004500}
}
```



### Installation

To install this repository, it is preferable to create a new conda environment. For example:

```bash
conda create -n your_environment_name python=3.9
conda activate your_environment_name
```

Then, use the `setup.py` to install the basic requirements for setting up the environment:

```bash
pip install .
# or
python setup.py install
```

### Synthetic Data

To generate synthetic data for the repressilator, Goodwin oscillator, or the SIR model, define the output file path, training, validation, and testing file names, the number of combinations, and the number of time series per combination. For example:

```bash
python Repressilator.py --output_path ./data --train_path training.hdf5 --valid_path validation.hdf5 --test_path testing.hdf5 -n 3 --num_samples 3000
```

### Repressilator Example

There are 5 different experiments for the embedding and the GPT + Embedding for the repressilator model based on the architecture of the embedding model used. For the embedding models, provide the experiment name, the training data path, and the testing data path. For the transformer model, provide the experiment name, the training and validation data path, and the embedding model checkpoint. The 5 different experiments used are MLP network, KAN network, MLP+KAN network (mKAN), eKdM (encoder KAN decoder MLP), and eMdK (encoder MLP decoder KAN).

#### MLP Network

To run the embedding model:

```bash
python train_Repressilator_enn.py --exp_name exp --train ./training_file.hdf5 --eval ./valid.hdf5
```

To run the transformer model:

```bash
python train_repressilator_transformer.py --exp_name exp --train ./training_file.hdf5 --eval ./valid.hdf5 --embedding_file_or_path ./embeddingrepressilator300.pth
```

#### KAN Network

To run the embedding model:

```bash
python train_repressilator_enn_eKAN.py --exp_name exp --train ./training_file.hdf5 --eval ./valid.hdf5
```

To run the transformer model:

```bash
python train_repressilator_transformer_eKAN.py --exp_name exp --train ./training_file.hdf5 --eval ./valid.hdf5 --embedding_file_or_path ./embeddingrepressilator300.pth
```

#### mKAN Network

To run the embedding model:

```bash
python train_repressilator_enn_mKAN.py --exp_name exp --train ./training_file.hdf5 --eval ./valid.hdf5
```

To run the transformer model:

```bash
python train_repressilator_transformer_mKAN.py --exp_name exp --train ./training_file.hdf5 --eval ./valid.hdf5 --embedding_file_or_path ./embeddingrepressilator300.pth
```

#### eKdM

To run the embedding model:

```bash
python train_repressilator_enn_eKdM.py --exp_name exp --train ./training_file.hdf5 --eval ./valid.hdf5
```

To run the transformer model:

```bash
python train_repressilator_transformer_eKdM.py --exp_name exp --train ./training_file.hdf5 --eval ./valid.hdf5 --embedding_file_or_path ./embeddingrepressilator300.pth
```

#### eMdK

To run the embedding model:

```bash
python train_repressilator_enn_eMdK.py --exp_name exp --train ./training_file.hdf5 --eval ./valid.hdf5
```

To run the transformer model:

```bash
python train_repressilator_transformer_eMdK.py --exp_name exp --train ./training_file.hdf5 --eval ./valid.hdf5 --embedding_file_or_path ./embeddingrepressilator300.pth
```

**Note:** To speed up training when using the GPT model, the dataset is transformed using the embedding model and cached for faster training. When trying different configurations for the GPT model using the same embedding model, the cached file name is the same for the MLP, KAN, and mKAN models. For the eKdM and eMdK models, the cached file name is also the same. It is advisable to remove the cached file when training different models.

### Evaluation for the Repressilator Models

To evaluate the embedding model based on different isotropic measures, Jaccard, and hit rate measures, and to measure the reconstruction and prediction rate, follow the steps in the `Embedding_Evaluation.ipynb` file. For comparison of the different embedding models and GPT models, use the steps in `Architectures_Evaluation.ipynb`.

