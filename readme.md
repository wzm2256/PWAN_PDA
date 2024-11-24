

-----------------------
#### Update:
Version 2 is comming soon! Faster, less parameter tuning, and better performance!

-----------------------



-----------------------
#### Personal update:
I am looking for a machine learning engineer/ machine learning researcher position in EU. If you know suitable opportunity, please do not hesitate to contact me! My linkedin page: [Ziming Wang](https://www.linkedin.com/in/ziming-wang-50856916a/)

-----------------------




# Partial Distribution Matching via Partial Wasserstein Adversarial Networks


This is the official implement of the partial domain adaptation method proposed in [Partial Distribution Matching via
Partial Wasserstein Adversarial Networks](https://arxiv.org/abs/2409.10499). 
This approach applies a PWAN model to partial domain adaptation problems.
Also see the application of PWAN in point set registration in our earlier [work](https://openreview.net/forum?id=2ggNjUisGyr) 
([Code](https://github.com/wzm2256/PWAN))

### How does it work?
A classifier is used to extract features and predict labels,
while a PWAN model is used to align the source features to a fraction of the reference features.

<img src="Readme_fig\PDA.png" width="356"/>

An example of the T-SNE visualization of the reference (blue) and source features (red).
Gray points represent outlier features.

| without PWAN                                 | With PWAN                                    |  
|----------------------------------------------|----------------------------------------------|
 <img src="Readme_fig\fig1.png" width="256"/> | <img src="Readme_fig\fig3.png" width="256"/> 


### Requirement
- torch=1.12
- matplotlib
- tqdm
- scikit-learn
- tensorboard

### Usage
#### Data perparation

1. **OfficeHome**
   1. Download the dataset from the [official website](https://www.hemanthdv.org/officeHomeDataset.html) to `data/OfficeHome`
   2. Rename the folder of ``real world``  to `Real_World` 

2. **ImageNet**: 
   1. Download the ImageNet dataset (`ILSVRC2012_img_train.tar`) from the [official website](https://www.image-net.org/download.php) to `data/ImageNetCaltech`.
   2. Uncompress the ImageNet dataset to a `train/` folder. 

3. **DomainNet** and **VisDa17** can be downloaded automatically. If the link expires, please also use the official websites.


#### Script
The main script is `PWANN.py`.
We also provide a convinient script `run_all.py` for running this script with different dataset and random seeds.
Commands for reproducing the results in the paper are provided in `a.txt`


The visualization and summary code is provided in `Plot` folder.
For example,

```
cd Plot
python vis_tensorboard.py ./LOG/OfficeHome
```
summarize all experiments in OfficeHome folder.




#### Results
There will be some randomness in the results due to random initialization.
We report the results with random seed 0,1,2.

**OfficeHome**:

| seed   | 0 | 1 | 2 |
|--------|-------|-------|----|
| AC	    |0.66 |0.66 |0.62| 
| AP	    |0.86 |0.81 |0.86| 
| AR	    |0.89 |0.90 |0.90| 
| CA	    |0.78 |0.77 |0.76| 
| CP	    |0.77 |0.74 |0.80| 
| CR	    |0.83 |0.84 |0.87| 
| PA	    |0.79 |0.75 |0.79| 
| PC	    |0.64 |0.63 |0.63| 
| PR	    |0.87 |0.86 |0.87| 
| RA	    |0.80 |0.79 |0.80| 
| RC	    |0.66 |0.66 |0.68| 
| RP	    |0.86 |0.86 |0.89| 
| Avg   |0.784|0.772|0.791|

**VisDa17**: The accuracy ranges from 80 to 92.
The large variance is due to the ambiguity of the `skateboard` and `knife` class.
Visualization and discussion can be found in the appendix of the paper.

**DomainNet**:

| seed   | 0      | 1      | 2     |
|--------|--------|--------|-------|
|CP	| 0.55	  | 0.54   | 0.51  |
|CR	| 0.73	  | 0.75   | 0.72  |
|CS	| 0.57	  | 0.61   | 0.56  |
|PC	| 0.65	  | 0.65   | 0.64  |
|PR	| 0.81	  | 0.82   | 0.81  |
|PS	| 0.72	  | 0.72   | 0.73  |
|RC	| 0.77	  | 0.78   | 0.77  |
|RP	| 0.72	  | 0.73   | 0.73  |
|RS	| 0.70	  | 0.70   | 0.70  |
|SC	| 0.52	  | 0.50   | 0.49  |
|SP	| 0.56	  | 0.54   | 0.54  |
|SR	| 0.66	  | 0.66   | 0.63  |
|Avg| 	0.668 | 	0.672 | 0.657 |


**ImageNetCaltech**:

| seed | 0 | 1 | 2 | Avg|
|------|-------|-------|----| ---|
| IC   |0.858 |0.864 |0.857| 0.860|






### Reference
If you find the code useful, please cite the following papers.

    @misc{wang2024partialdistributionmatchingpartial,
          title={Partial Distribution Matching via Partial Wasserstein Adversarial Networks}, 
          author={Zi-Ming Wang and Nan Xue and Ling Lei and Rebecka Jörnsten and Gui-Song Xia},
          year={2024},
          eprint={2409.10499},
          url={https://arxiv.org/abs/2409.10499}, 
    }

    @inproceedings{wang2022partial,
        title={Partial Wasserstein Adversarial Network for Non-rigid Point Set Registration},
        author={Zi-Ming Wang and Nan Xue and Ling Lei and Gui-Song Xia},
        booktitle={International Conference on Learning Representations (ICLR)},
        year={2022}
    }



For any question, welcome to open an issue or contact me.

### Acknowledgement
The `datasets` module was adopted from the [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library).
We thank the authors of this repository and other authors in the community for their code.


### LICENSE
The code is available under a [MIT license](LICENSE).


