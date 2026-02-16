---
sidebar_position: 30
---

# Citation

If you use SpikeSEG in your research, please cite the relevant publications.

## Primary Citations

### STDP Learning Rule and SNN Architecture

```bibtex
@article{kheradpisheh2018stdp,
  title     = {{STDP}-based spiking deep convolutional neural networks for object recognition},
  author    = {Kheradpisheh, Saeed Reza and Ganjtabesh, Mohammad and Thorpe, Simon J. and Masquelier, Timoth{\'e}e},
  journal   = {Neural Networks},
  volume    = {99},
  pages     = {56--67},
  year      = {2018},
  publisher = {Elsevier},
  doi       = {10.1016/j.neunet.2017.12.005}
}
```

### SpikeSEG Encoder-Decoder

```bibtex
@inproceedings{kirkland2020spikeseg,
  title        = {{SpikeSEG}: Spiking segmentation via {STDP} saliency mapping},
  author       = {Kirkland, Paul and Di Caterina, Gaetano and Soraghan, John and Andreopoulos, Yiannis and Matich, George},
  booktitle    = {2020 IEEE International Joint Conference on Neural Networks (IJCNN)},
  pages        = {1--8},
  year         = {2020},
  organization = {IEEE},
  doi          = {10.1109/IJCNN48605.2020.9207100}
}
```

### HULK-SMASH Instance Segmentation

```bibtex
@article{kirkland2022hulksmash,
  title   = {Unsupervised spiking instance segmentation on event data using {STDP} features},
  author  = {Kirkland, Paul and Di Caterina, Gaetano and Soraghan, John and Andreopoulos, Yiannis and Matich, George},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  year    = {2022},
  doi     = {10.1109/TNNLS.2022.3185182}
}
```

### Space Domain Awareness (IGARSS 2023)

```bibtex
@inproceedings{kirkland2023igarss,
  title        = {Neuromorphic sensing and processing for space domain awareness},
  author       = {Kirkland, Paul and others},
  booktitle    = {IGARSS 2023 -- 2023 IEEE International Geoscience and Remote Sensing Symposium},
  year         = {2023},
  organization = {IEEE},
  doi          = {10.1109/IGARSS52108.2023.10283393}
}
```

## Dataset

### EBSSA

```bibtex
@article{afshar2020ebssa,
  title   = {Event-based object detection and tracking for space situational awareness},
  author  = {Afshar, Saeed and Nicholson, Andrew P. and van Schaik, Andre and Cohen, Gregory},
  journal = {IEEE Sensors Journal},
  volume  = {20},
  number  = {24},
  pages   = {15117--15132},
  year    = {2020},
  doi     = {10.1109/JSEN.2020.3009687}
}
```

### N-MNIST

```bibtex
@article{orchard2015nmnist,
  title   = {Converting static image datasets to spiking neuromorphic datasets using saccades},
  author  = {Orchard, Garrick and Jayawant, Ajinkya and Cohen, Gregory K. and Thakor, Nitish},
  journal = {Frontiers in Neuroscience},
  volume  = {9},
  pages   = {437},
  year    = {2015},
  doi     = {10.3389/fnins.2015.00437}
}
```

## Implementation Disclaimer

This is an **independent PyTorch implementation** of the SNN-based satellite detection pipeline described in the publications listed above (Kirkland et al., Kheradpisheh et al.). **No source code was obtained from the original authors** — the implementation was developed entirely from first principles based on the published methodology, equations, and architectural descriptions contained in the cited papers.

This implementation serves as **Type 1 Compute's internal capability** for event-based space domain awareness. It is not affiliated with, endorsed by, or derived from any code belonging to the original authors or their institutions.

## Acknowledgments

We gratefully acknowledge the foundational research of Kheradpisheh et al. and Kirkland et al., whose published work made this independent implementation possible. The EBSSA dataset is provided by the International Centre for Neuromorphic Systems (ICNS) at Western Sydney University.

For questions or issues, please open a [GitHub issue](https://github.com/type1compute/SPIKESEG--Spiking-Neural-Network-for-Event-Based-Space-Domain-Awareness/issues).
