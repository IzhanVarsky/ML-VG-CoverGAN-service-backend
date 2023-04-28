# CoverGAN backend service

The backend server is running on port `1123`.
Service is available at http://109.188.135.85:5001/covergan.

## Weights

* The pretrained weights can be downloaded
  from [here](https://drive.google.com/file/d/1ArU0TziLBOxhphG4KBshUxPBBECErxu1/view?usp=sharing)
* These weights should be placed into `/covergan/weights` folder

## Running using Docker

In this service two types of generator are available:

* The first one creates the covers with abstract lines
* The second one draws closed forms.

It is also possible to use one of two algorithms for applying inscriptions to the cover:

* The first algorithm uses the captioner model
* The second is a deterministic algorithm which searches for a suitable location

The service uses pretrained weights. See [this](README.md#Weights) section.

### Building

* Specify PyTorch version to install in [`Dockerfile`](./Dockerfile).

* Build the image running `docker_build_covergan_service.sh` file

### Running

* Start the container running `docker_run_covergan_service.sh` file

## License

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/

[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png

[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg