# Neural Style Transfer 

This is a PyTorch implementation of neural style transfer based on the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576) by Gatys et al. 
The aim is to transfer the artistic style of one image onto the content of another using a deep CNN (in this case a pre-trained VGG19 network).

Here is an example of the Taj Mahal mixed with The Starry Night by Vincent van Gogh:

<div align="center">
    <img src="images/style/starrynight.jpg" alt="Starry Night" width="256"/>
    <img src="images/content/tajmahal.jpg" alt="Taj Mahal" width=256/>
    <img src="results/tajmahal_styled_with_starrynight_20251223-194256/result.jpg" alt="Starry Taj Mahal" width="512"/>
</div>

---

Here are a couple more examples:

<p align="center">
<img src="results/pytorch-pretrained-models/candy_man.jpg" width="270px">
<img src="results/pytorch-pretrained-models/rain_man.jpg" width="270px">
<img src="results/man_styled_with_mosaic_20251223-222931/result.jpg" width="270px">

<img src="results/pytorch-pretrained-models/candy_taj_mahal.jpg" width="270px">
<img src="results/pytorch-pretrained-models/tajmahal_rain_princess.jpg" width="270px">
<img src="results/pytorch-pretrained-models/mosaic_taj_mahal.jpg" width="270px">
</p>

---

And here are some results coupled with their style:

<p align="center">
<img src="results/gray_bridge_styled_with_vg_la_cafe_20251224-181051/result.jpg" height="267px">
<img src="images/style/vg_la_cafe.jpg" height="267px">
<br><br>
    
<img src="results/gray_bridge_styled_with_wave_crop_20251224-181717/result.jpg" height="267px">
<img src="images/style/wave_crop.jpg" height="267px">
<br><br>

<img src="results/pytorch-pretrained-models/rain_robot.jpg" height="300px">
<img src="images/style/rain-princess.jpg" height="300px">
<br><br>

<img src="results/golden_gate_styled_with_sunflowers_20251223-230124/result.jpg" height="300px">
<img src="images/style/sunflowers.jpg" height="300px">
<br><br>

<img src="results/ronaldo_styled_with_ben_giles_20251223-225458/result.jpg" height="300px">
<img src="images/style/ben_giles.jpg" height="300px">
</p>

---

I have also optimized this naive implementation following [Perceptual Losses for Real-Time Style Transfer
and Super-Resolution](https://arxiv.org/pdf/1603.08155) by Johnson et al., achieving significantly faster inference which enables the algorithm to be applied to videos too!

<p align="center">
    <img src="gifs/monkey.gif" width="300" title="Monkey">
    <img src="gifs/monkey_candy.gif" width="300" title="Candy monkey">
    <br><br>
    <img src="gifs/swans.gif" width="300" title="Swans">
    <img src="gifs/swans_mosaic.gif" width="300" title="Mosaic swans">
    <br><br>
    <img src="gifs/tiger.gif" width="300" title="Tiger">
    <img src="gifs/tiger_rain_princess.gif" width="300" title="Rain princess tiger">
</p>

---

# How To Use

### Clone The Repository And Set Up The Environment

```bash
git clone git@github.com:ikemal12/Neural-Style-Transfer.git
cd Neural-Style-Transfer
```

It is recommended to use a virtual environment:

```bash
python -m venv .venv
source .venv/Scripts/activate  # On Windows
# or
source .venv/bin/activate      # On macOS/Linux
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### Prepare Images

Place your content images in `images/content/` and style images in `images/style/`, or use the ones already present.

### Style Your Own Image

To style your own image, you can run:

```bash
python neural_style_transfer.py --content PATH_TO_CONTENT_IMAGE --style PATH_TO_STYLE_IMAGE --output PATH_TO_OUTPUT_IMAGE
```

You can choose not to specify an output path in which case the output image will automatically be saved in a timestamped folder in `results/`.
For example, to generate the taj mahal with the style of starry night, you can run:

```bash
python neural_style_transfer.py --content images/content/tajmahal.jg --style images/style/starrynight.jpg 
```

If you want a much faster result (albeit slightly lower quality), you can run with a pretrained model:

```bash
python inference.py --content PATH_TO_CONTENT_IMAGE --model PATH_TO_MODEL --output results/your_result.jpg
```

Note the output path must start with the `results/` folder. For example:

```bash
python inference.py --content images/content/man.jpg --model models/candy.pth --output results/candy_man.jpg
```

### Style Your Own Video

To style your own video, you can run:

```bash
python video_style_transfer.py --input PATH_TO_INPUT_VIDEO --model PATH_TO_MODEL --output PATH_TO_OUTPUT_VIDEO
```

For example:

```bash
python video_style_transfer.py --input videos/tiger.mp4 --model models/rain_princess.pth --output results/tiger_rain_princess.mp4
```

You can use any pretrained model in the `models/` directory, or train and use your own model which is covered below. 

### Train Your Own Model

To train your own style transfer model, use the train.py script. Example usage:

```bash
python train.py --content PATH_TO_CONTENT_IMAGE --style PATH_TO_STYLE_IMAGE --output_dir results/
```

You can customize the training with additional arguments:

```
usage: train.py [-h] --content CONTENT --style STYLE [--output OUTPUT] [--output_dir OUTPUT_DIR]
                                [--imsize IMSIZE] [--num_steps NUM_STEPS] [--style_weight STYLE_WEIGHT]
                                [--content_weight CONTENT_WEIGHT] [--init_random]

Neural Style Transfer

options:
    -h, --help            show this help message and exit
    --content CONTENT     Path to the content image
    --style STYLE         Path to the style image
    --output OUTPUT       Output file path (e.g., results/output.jpg)
    --output_dir OUTPUT_DIR
                                                Base output directory (used if --output not specified)
    --imsize IMSIZE       Size of output image
    --num_steps NUM_STEPS
                                                Number of optimization steps
    --style_weight STYLE_WEIGHT
                                                Weight for style loss
    --content_weight CONTENT_WEIGHT
                                                Weight for content loss
    --init_random         Initialize with random noise (default: content image)
```

