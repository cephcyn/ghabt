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
import motivationalLeo1 from './../img/motivational-leo-v1.png'
import motivationalLeo2 from './../img/motivational-leo-v2.png'
import modelchart from './../img/model_chart.png'
import statsImage from './../img/stats-image.png'
import statsLanguage from './../img/stats-language.png'
import exampleAwwnime from './../img/example-awwnime.png'
import exampleTumblr from './../img/example-tumblr.png'
import exampleDogelore from './../img/example-dogelore.png'

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

  const [imgLeo, setImgLeo] = useState(motivationalLeo1)
  const toggleLeo = (event) => {
    if (imgLeo===motivationalLeo1) {
      setImgLeo(motivationalLeo2);
    } else {
      setImgLeo(motivationalLeo1);
    }
  };

  return (
    <Container className={classes.root}>
      <CssBaseline />
      <Paper className={`${classes.panel} ${classes.shinypanel}`} style={{ textAlign:'center' }}>
        <Typography variant="h1">
          GANna Have A Bad Time
        </Typography>
        <Typography variant="h4">
          Which model architectures stand up the best to GANs?
        </Typography>
      </Paper>
      <Paper className={classes.panel}>
        <Typography>
          TODO: Relatively informal description of the motivation of the project (equate this to the "Problem Setup" of the video) <em> Make this the easy-to-read description, one paragraph MAX. </em>
        </Typography>
      </Paper>
      <Paper className={classes.detailtext}>
        <Typography variant="h4" style={{ textAlign:'center' }} gutterBottom>
          Abstract
        </Typography>
        <Typography>
          TODO Write an abstract / outline of what we actually did
        </Typography>
      </Paper>
      <Paper className={classes.panel} style={{ textAlign:'center' }}>
        <AspectRatio ratio="16 / 9" style={{ maxWidth: '60%', left: '50%', transform: 'translate(-50%, 0)' }}>
          <iframe
            src="https://www.youtube-nocookie.com/embed/E2xNlzsnPCQ" // <!-- TODO this needs to be a 2-3min video covering problem setup, data used, techniques used (for both of those latter need to distinguish what is new and what isnt) -->
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen>
          </iframe>
        </AspectRatio>
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
            TODO write a paragraph for the problem setup
          </Typography>
        </Paper>
        <Paper className={classes.detailtext}>
          <Typography variant="h5" style={{ textAlign:'center' }} gutterBottom>
            Data Used
          </Typography>
          <Typography gutterBottom>
            TODO describe datasets that we used that were not original (i.e. code from github)
          </Typography>
          <Typography gutterBottom>
            TODO describe datasets that we used that were original (i.e. new code, gathered dataset, etc), if we didn't have any then explicitly say that
          </Typography>
        </Paper>
        <Paper className={classes.detailtext}>
          <Typography variant="h5" style={{ textAlign:'center' }} gutterBottom>
            Techniques Used
          </Typography>
          <Typography gutterBottom>
            TODO describe techniques that we used that were not original (i.e. code from github)
          </Typography>
          <Typography gutterBottom>
            TODO describe techniques that we used that were original (i.e. new code, gathered dataset, etc), if we didn't have any then explicitly say that
          </Typography>
        </Paper>
        <Paper className={classes.detailtext}>
          <Typography variant="h5" style={{ textAlign:'center' }} gutterBottom>
            Experiments and Results
          </Typography>
          <Typography gutterBottom>
            TODO write a paragraph for related work if there's any that we reference, otherwise remove this card
          </Typography>
        </Paper>
      </Paper>
      <Paper className={`${classes.panel} ${classes.shinypanel}`}>
        <Container style={{ width:"40%" }}>
          <Image
            src={imgLeo}
            alt="Leo DiCaprio numpy meme (credits: Will Chen)"
            color="transparent"
            onClick={toggleLeo}
            style={{ height:"100px" }}
          />
        </Container>
      </Paper>
    </Container>
  );
}
