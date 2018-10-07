# AML-project

Instructions on how to run models

Requirements: Python 3.6, Pytorch 0.4

    Prepare repository and dataset

`git clone https://github.com/conrad784/AML-project.git`

`$ AML-project/PyTorch-GAN/data/download_cyclegan_dataset.sh maps`

    Run one of the models:

`cd AML-project/PyTorch-GAN/implementations/`

CycleGAN
`python cyclegan/cyclegan.py --dataset_name maps --img_width 256 --img_height 256 --sample_interval 100 --checkpoint_interval 5`

Unit
`python unit/unit.py --dataset_name maps --img_width 256 --img_height 256 --sample_interval 100 --checkpoint_interval 5`

InvAuto
`python invauto/invauto.py --dataset_name maps --img_width 128 --img_height 128 --sample_interval 100 --checkpoint_interval 5`
