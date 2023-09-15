# SDMask2Former
The official PyTorch implementation of diffusion model as an image encoder, combined with Mask2Former's pixel decoder and transformer decoder 
for referring expression segmentation.

## Framework

Detailed architecture refer to [Mask2Former](https://github.com/facebookresearch/Mask2Former).

## Preparation

1. Environment
   - [PyTorch](www.pytorch.org)
   - [Stable Diffusion dependencies](https://github.com/CompVis/stable-diffusion)
2. Datasets
   - The detailed instruction is in [LAVT](https://github.com/yz93/LAVT-RIS).
3. Pretrained weights
   - refer to [ODISE](https://github.com/NVlabs/ODISE)

## Train and Test

Refer to [LAVT](https://github.com/yz93/LAVT-RIS).

## License

This project is under the MIT license. See [LICENSE](LICENSE) for details.


Some code changes come from [CRIS](https://github.com/DerrickWang005/CRIS.pytorch/tree/master) and [LAVT](https://github.com/yz93/LAVT-RIS).
