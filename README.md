<figure>
  <img src="/other_files/repo_info.png" alt="example" width="100%"/>
</figure> <br>

## Introduction

This is a repository to accompany 'Unveiling the Critical Nexus of Data Preprocessing and Transparent Documentation for Result Quality and Reproducibility in Digital History', forthcoming in Digital Humanities Quarterly (DHQ). It contains the code for the case study section of the paper where we explore the importance of adequate data preprocessing, transparency, and documentation in digital history research, showcasing how these often overlooked practices impact research quality and reproducibility.


More information is available at the project website: [projectmape.org](https://www.projectmape.org)

##  publications
* Digital Humanities Quarterly (DHQ): TBA
* Journal of Social Networks:  [Networks from archives: Reconstructing networks of official correspondence in the early modern Portuguese empire](https://doi.org/10.1016/j.socnet.2020.08.008)


## How do I get set up? ###

Clone or download the repository. The list of dependencies is listed in the requirements file and can be installed via the following Python command: 
```
pip install -r requirements.txt
```

Alternatively, you can create an environment with the configured to the project:
```
conda env create -f environment.yml
``` 

The dataset used is currently being prepared for public release; please get in touch with the authors for early access. The custom NER model can also be shared upon request. The GSDMM implementation was based on the code provided by Yin and Wang and is available in this [repository](https://github.com/rwalk/gsdmm). 

## Who do I talk to? ###

* Overal questions about the project can be directed to it's pricipal inverstigators via our [website](https://www.projectmape.org/contact).
* Although we tried out best to provide accurate implementation of the methods, if you find errors or something is not clear, please send a mesage to Clodomir Santana (clodomir@ieee.org)

## License

This work is free. You can redistribute it and/or modify it under the terms of the GNU Public license and subject to all prior terms and licenses imposed by the free, public data sources provided by the [PT-AHU](https://digitarq.ahu.arquivos.pt), i.e. the 'data originators'. The code comes without any warranty, to the extent permitted by applicable law.
