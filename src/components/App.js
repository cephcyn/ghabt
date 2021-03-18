import React, { useState } from 'react';
import Container from '@material-ui/core/Container';
import CssBaseline from '@material-ui/core/CssBaseline';
import Typography from '@material-ui/core/Typography';
import Paper from '@material-ui/core/Paper';
import Button from '@material-ui/core/Button';
import { makeStyles } from '@material-ui/core/styles';

import 'react-aspect-ratio/aspect-ratio.css'
import AspectRatio from 'react-aspect-ratio';
import Carousel from 'react-material-ui-carousel'
import Image from 'material-ui-image'

import "fontsource-roboto"
import previous_work from './../img/previous_work.png'
import expr_results from './../img/455_results.PNG'
import bird_ex from './../img/bird_pert.PNG'
import ship_ex from './../img/ship_ex.PNG'

import ModelDemo from './ModelDemo'

const useStyles = makeStyles((theme) => ({
  root: {
    align: 'center',
  },
  panel: {
    padding: '15px 30px',
    marginTop: '20px',
    marginBottom: '20px',
  },
  examplecard: {
    padding: '10px 30px',
    position: 'relative',
    left: '50%',
    transform: 'translate(-50%, 0)',
    maxWidth: '80%',
  },
  memetitletext: {
    fontFamily: 'Comic Sans MS, Comic Sans, Comic Neue, cursive',
  },
  detailtext: {
    padding: '10px 30px',
    marginTop: '15px',
    marginBottom: '15px',
  },
  shinybutton: {
    background: 'linear-gradient(45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab)',
    backgroundSize: '400% 400%',
    animation: '$gradient 15s ease infinite',
    border: 0,
    borderRadius: 3,
    boxShadow: '0 3px 5px 2px rgba(255, 105, 135, .5)',
    color: 'white',
    height: 48,
    padding: '0 30px',
  },
  shinypanel: {
    background: 'linear-gradient(45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab)',
    backgroundSize: '400% 400%',
    animation: '$gradient 15s ease infinite',
  },
  '@keyframes gradient': {
  	'0%': {
  		'background-position': '0% 50%',
  	},
  	'50%': {
  		'background-position': '100% 50%',
  	},
  	'100%': {
  		'background-position': '0% 50%',
  	},
  }
}));

export default function App() {
  const classes = useStyles();

  return (
    <Container className={classes.root}>
      <CssBaseline />
      <Paper className={`${classes.panel} ${classes.shinypanel}`} style={{ textAlign:'center' }}>
        <Typography variant="h1">
          Bullying Models with Perturbation
        </Typography>
        <Typography variant="h4">
          Which model architectures stand up the best to adversarial image perturbation?
        </Typography>
      </Paper>
      <Paper className={classes.panel}>
        <Typography variant="h4" style={{ textAlign:'center' }} gutterBottom>
          Problem Description
        </Typography>
        <Typography gutterBottom>
          In recent decades, neural networks have had a large role in the area of computer vision. Computer vision has recently garnered increasing international attention as it has been applied to the world of surveillance. As a reaction to this, individuals with privacy concerns have increasingly become more interested in how to trick these networks into labeling images incorrectly. In particular, the most popular attacks involve changing what the model would see in small enough ways to trick the model, but still be identifiable to humans. <b>We tested how resistant various neural network architectures would be to a whitebox adversarial attack via image perturbation. We also tested how well using one round of image perturbation as a data augmentation method works to improve model performance</b>.
        </Typography>
        <Typography gutterBottom>
          Our approach was to investigate a variety of neural network architectures ranging from linear models to the original ResNet model. One of the most common benchmark datasets is the CIFAR-10 dataset, which we used to train and identify the most accurate neural network architectures out of a few broad categories. From here, we devised a way to perturb images via a whitebox adversarial attack on the trained models. This attack consists of taking a model trained on CIFAR-10, and learning how to perturb images by going against the gradient. We then output perturbed images, and then compare how well these models are able to correctly label the images. We found that more complex models tended to perform better on the CIFAR data set, and that as the models increased in complexity, they became increasingly susceptible to our adversarial attack but were better able to learn from it if we trained them using data perturbed in a similar way.
        </Typography>
      </Paper>
      <Paper className={classes.panel} style={{ textAlign:'center' }}>
        <AspectRatio ratio="16 / 9" style={{ maxWidth: '60%', left: '50%', transform: 'translate(-50%, 0)' }}>
          <iframe
            src="https://www.youtube-nocookie.com/embed/W-Ak12sdHdo" // <!-- TODO this needs to be a 2-3min video covering problem setup, data used, techniques used (for both of those latter need to distinguish what is new and what isnt) -->
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen>
          </iframe>
        </AspectRatio>
      </Paper>
      <Paper className={classes.panel}>
        <Typography variant="h4" style={{ textAlign:'center' }} gutterBottom>
          Previous Work
        </Typography>
        <Typography gutterBottom>
          Our original project proposal was inspired by <a href="https://arxiv.org/abs/2002.02196">AI-GAN: Attack-Inspired Generation of Adversarial Examples (Bai et al, 2020)</a>. We wanted to create a GAN that would create perturbed adversarial images in response to a CIFAR-10 trained neural network. However, we found that this would require much more work than originally planned (see further discussion below), so our project shifted away from using GANs to instead generating perturbed images with a whitebox attack.
        </Typography>
        <Container style={{ textAlign:'center', width:'50%' }}>
            <Image
              src={previous_work}
              alt="previous work table"
              aspectRatio={3/1}
            />
        </Container>
        <Typography gutterBottom>
          Another relevant work is <a href="https://arxiv.org/pdf/1801.00553.pdf">Threat of Adversarial Attacks on Deep Learning in Computer Vision: A Survey (Akhtar and Mian, IEEE Access 2018)]</a> which examined different methods of performing adversarial attacks on deep learning models with a wide range of approaches (image above related). To contrast with this, in our experiment, we use a uniform attack style but primarily vary the architecture of the model being attacked.
        </Typography>
        <Typography gutterBottom>
          For our experiment itself, we used the <a href="https://www.cs.toronto.edu/~kriz/cifar.html">pre-existing CIFAR-10 dataset</a> and a <a href="https://pytorch.org/vision/stable/models.html">pretrained ResNet</a> model.
        </Typography>
      </Paper>
      <Paper className={classes.panel}>
        <Typography variant="h4" style={{ textAlign:'center' }} gutterBottom>
          Behind The Scenes
        </Typography>
        <Paper className={classes.detailtext}>
          <Typography variant="h5" style={{ textAlign:'center' }} gutterBottom>
            Problem Setup
          </Typography>
          <Typography gutterBottom>
            Our first step was to identify and build models. We decided to try four different model architecture types: a linear model, a convolutional model, a convolutional model with batch norm/dropout, and finally a high performance model called ResNet. These architectures were investigated, trained with the CIFAR dataset, and tweaked to improve their accuracy. The most accurate version of each of these architectures was then selected to be used in our attack experiment. 
          </Typography>
          <Typography gutterBottom>
            Once we had our trained models, we used them to perturb images by calculating the gradient at the input level and adding it to an input perturbation vector. The input-level gradient allows us to note which features would lead the model to label images the way they did, so we use this to ideally perturb images so that these important features wouldn't show up. Finally, we saved the perturbed images generated from each model to create both a perturbed training and testing dataset.
          </Typography>
          <Typography gutterBottom>
            After generating our perturbed image datasets, we tested them on our architectures. For each architecture, we trained models in two different ways and compared each trained model’s performance on the original and perturbed image datasets. The first way we trained each model type was with 50 epochs on the original CIFAR training data. The second way we trained each model type was with 25 epochs on the combined (original CIFAR and perturbed) training data. These variants were tested on the perturbed testing data and the original testing data. Finally, we compared the difference in accuracy of these two models across architecture groups.

          </Typography>
        </Paper>
        <Paper className={classes.detailtext}>
          <Typography variant="h5" style={{ textAlign:'center' }} gutterBottom>
            Data Used
          </Typography>
          <Typography gutterBottom>
            The main dataset that we used was CIFAR-10.
          </Typography>
          <Typography gutterBottom>
            The other datasets we used were those of perturbed images generated during the experiment, which were derived from CIFAR-10.
          </Typography>
        </Paper>
        <Paper className={classes.detailtext}>
          <Typography variant="h5" style={{ textAlign:'center' }} gutterBottom>
            Techniques Used
          </Typography>
          <Typography gutterBottom>
            We used PyTorch for constructing our models. We also tuned our ResNet model from the pretrained model PyTorch provides. Information on the ResNet model can be found <a href="https://pytorch.org/vision/stable/models.html">here</a>.
          </Typography>
          <Typography gutterBottom>
            Other than those, our model architecture and perturbation code was written ourselves.
          </Typography>
        </Paper>
        <Paper className={classes.detailtext}>
          <Typography variant="h5" style={{ textAlign:'center' }} gutterBottom>
            Experiments and Results
          </Typography>
          <Container style={{ textAlign:'center', width:'50%' }}>
              <Image
                src={expr_results}
                alt="expereiment results"
                aspectRatio={4/2}
              />
          </Container>
          <Typography gutterBottom>
            Overall, we found that more complex models tended to perform better on the CIFAR data set, as expected. One thing that was surprising about this was that as the models increased in complexity, they became increasingly susceptible to our adversarial attack. Interestingly, while the more complex models were less resilient against image perturbation attacks, they did learn to handle perturbed data more adeptly when we trained them on an augmented dataset. For all model architectures we tested, augmenting the input dataset with the perturbed images decreased the final model performance on the original CIFAR test set, but increased performance on the perturbed test set. This could be because the perturbations serve a kind of “specialized” data augmentation to help the model generalize better to images with a small amount of noise. In this sense, training on this augmented data set for a longer period of time may lead to better overall performance in the long run.
          </Typography>
          <Container style={{ textAlign:'center', width:'50%' }}>
              <Image
                src={bird_ex}
                alt="example of a perturbed bird image"
                aspectRatio={1.5/1}
              />
          </Container>
          <Container style={{ textAlign:'center', width:'50%' }}>
              <Image
                src={ship_ex}
                alt="example of a perturbed ship image"
                aspectRatio={2/1}
              />
          </Container>
        </Paper>
      </Paper>
      <Paper className={classes.panel}>
        <Typography variant="h4" style={{ textAlign:'center' }} gutterBottom>
          Discussion
        </Typography>
        <Paper className={classes.detailtext}>
          <Typography variant="h5" style={{ textAlign:'center' }} gutterBottom>
            Problems Encountered
          </Typography>
          <Typography gutterBottom>
            The first problem that we encountered encouraged a shift in our overall project approach. We had originally planned on using a GAN to generate adversarial perturbed images to attack our model, but when we tried this, we noticed how the GAN would create adversarial datasets containing images that don't match their label. Resolving this issue would involve a much larger code base than we believed we could develop, so we had to shift our project to a whitebox adversarial attack without the use of GAN's.
          </Typography>
          <Typography gutterBottom>
            Other than this, most problems we encountered were small bugs or difficulties with PyTorch and Google Colab. In particular, we repeatedly had issues with the models not training at all or having frustratingly long runtimes for training.
          </Typography>
        </Paper>
        <Paper className={classes.detailtext}>
          <Typography variant="h5" style={{ textAlign:'center' }} gutterBottom>
            Next Steps
          </Typography>
          <Typography gutterBottom>
            If we had more computational resources, we would explore more neural network architectures and see how those are affected by our adversarial attack. Alternatively, we could build on our current analysis by using visualizations to try to understand what accounts for the difference in impacts of image perturbation.
          </Typography>
          <Typography gutterBottom>
            We would also be interested in further exploring the idea of using GAN's and performing a black box adversarial attack.
          </Typography>
          <Typography gutterBottom>
            Finally, we would be interested in evaluating how well different neural network architectures respond to a more closed-loop GAN-based model training process.
          </Typography>
        </Paper>
        <Paper className={classes.detailtext}>
          <Typography variant="h5" style={{ textAlign:'center' }} gutterBottom>
            How our approach differs from others
          </Typography>
          <Typography gutterBottom>
            Our approach was a study of the various components that exist in modern computer vision deep neural networks and how they are affected by imager perturbations. 
          </Typography>
          <Typography gutterBottom>
            Compared to other studies, we examined the architecture of the model being attacked instead of the architecture of the attack or method of generating adversarial examples itself. This means we address an aspect of adversarial input attacks that other surveys have not examined in detail, which is overall beneficial. However, as mentioned in our discussion of future steps, it would be very interesting to do a multi-dimensional analysis studying interaction of different attack methods and attacked model architectures together as well.
          </Typography>
        </Paper>
      </Paper>
      <Paper className={classes.panel}>
        <Typography gutterBottom>
	  Check out our GitHub repo at <a href="https://github.com/cephcyn/ghabt">https://github.com/cephcyn/ghabt</a>!!
	</Typography>
      </Paper>
    </Container>
  );
}
