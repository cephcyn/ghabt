(this["webpackJsonpganna-have-a-bad-time"]=this["webpackJsonpganna-have-a-bad-time"]||[]).push([[0],{106:function(e,t,a){},153:function(e,t,a){"use strict";a.r(t);var n=a(2),o=a(0),i=a.n(o),r=a(10),s=a.n(r),l=(a(106),a(19)),c=a(200),d=a(203),h=a(62),m=a(161),u=a(202),p=a(74),g=(a(107),a(86)),f=a.n(g),b=a(81),w=a.n(b),x=a(35),y=a.n(x),j=(a(129),a.p+"static/media/motivational-leo-v1.ef976b8d.png"),v=a.p+"static/media/motivational-leo-v2.7c71dc7b.png",O=a.p+"static/media/model_chart.56d9d808.png",k=a.p+"static/media/stats-image.5285c4b1.png",B=a.p+"static/media/stats-language.e8c3434a.png",T=a.p+"static/media/example-awwnime.c9577563.png",A=a.p+"static/media/example-tumblr.88a44889.png",I=a.p+"static/media/example-dogelore.7de12bf1.png";a(15),a(33),a(201),a(204),a(61),a(83),a(82),Object(p.a)({icon:{width:"50%",height:"50%",color:"grey"}}),a(87),a(194),Object(p.a)({card:{width:"299px",height:"299px",position:"relative",display:"flex",alignItems:"center",justifyContent:"center",marginBottom:10},canvas:{width:"299px",height:"299px",zIndex:0,position:"absolute"},input:{zIndex:9999,position:"absolute"}});a(205),a(195),a(196),a(197),a(198),a(199),Object(p.a)({card:{height:"auto"},item:{paddingTop:10}});var R=a(34);Object(R.e)(),Object(p.a)((function(e){return{root:{padding:"15px 30px"},demoElement:{marginTop:"10px",marginBottom:"10px",width:"100%"},submit:{background:"linear-gradient(45deg, #d08771, #c85b85)",backgroundSize:"200% 200%",border:0,borderRadius:3,boxShadow:"0 3px 5px 2px rgba(255, 105, 135, .1)",color:"white",height:48,padding:"0 30px"},shiny:{background:"linear-gradient(45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab)",backgroundSize:"400% 400%",animation:"$gradient 15s ease infinite",boxShadow:"0 3px 5px 2px rgba(255, 105, 135, .5)"},"@keyframes gradient":{"0%":{"background-position":"0% 50%"},"50%":{"background-position":"100% 50%"},"100%":{"background-position":"0% 50%"}}}}));var N=Object(p.a)((function(e){return{root:{align:"center"},panel:{padding:"15px 30px",marginTop:"20px",marginBottom:"20px"},examplecard:{padding:"10px 30px",position:"relative",left:"50%",transform:"translate(-50%, 0)",maxWidth:"80%"},memetitletext:{fontFamily:"Comic Sans MS, Comic Sans, Comic Neue, cursive"},detailtext:{padding:"10px 30px",marginTop:"15px",marginBottom:"15px"},shinybutton:{background:"linear-gradient(45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab)",backgroundSize:"400% 400%",animation:"$gradient 15s ease infinite",border:0,borderRadius:3,boxShadow:"0 3px 5px 2px rgba(255, 105, 135, .5)",color:"white",height:48,padding:"0 30px"},shinypanel:{background:"linear-gradient(45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab)",backgroundSize:"400% 400%",animation:"$gradient 15s ease infinite"},"@keyframes gradient":{"0%":{"background-position":"0% 50%"},"50%":{"background-position":"100% 50%"},"100%":{"background-position":"0% 50%"}}}}));function F(){var e=N(),t=Object(o.useState)(j),a=Object(l.a)(t,2),i=a[0],r=a[1];return Object(n.jsxs)(c.a,{className:e.root,children:[Object(n.jsx)(d.a,{}),Object(n.jsxs)(m.a,{className:"".concat(e.panel," ").concat(e.shinypanel),style:{textAlign:"center"},children:[Object(n.jsx)(h.a,{variant:"h1",className:e.memetitletext,children:"MemeNet"}),Object(n.jsx)(h.a,{variant:"h4",className:e.memetitletext,children:"Multimodal Models Make Meme Market Manageable"})]}),Object(n.jsx)(m.a,{className:e.panel,children:Object(n.jsxs)(h.a,{children:['Artificial Intelligence (A.I.) has been applied in areas such as economics and algorithmic trading to great effect. In recent decades, the rise of viral Internet culture has led to the development of a new global economy: the online "meme economy". Drawing from scarce resources (such as creativity, humor, and time), individual producers (meme makers) offer their goods (memes in the form of multimodal ideas) over a centralized marketplace (Internet forums such as subreddits on Reddit) in exchange for currency (Internet points such as Upvotes or Likes). Oftentimes, knowing ',Object(n.jsx)("em",{children:"where"}),' to post a meme can greatly affect how well it is received by the Internet community. Posting in a highly apt channel can lead to instant Internet fame, while posting in a suboptimal channel can lead to one\'s creative work failing to gain attention, or worse, being stolen and reposted by meme thieves. Additionally, posting the same content in several different channels can be considered "spamming" and is negatively regarded. To make this decision easier for the millions of meme creators on the Internet, ',Object(n.jsx)("strong",{children:"we developed a multimodal neural network to predict the single best subreddit that a given meme should be posted to for maximum profit"}),"."]})}),Object(n.jsxs)(m.a,{className:e.detailtext,children:[Object(n.jsx)(h.a,{variant:"h4",style:{textAlign:"center"},gutterBottom:!0,children:"Abstract"}),Object(n.jsxs)(h.a,{children:["Deep neural networks are excellent at learning from data that consists of single modalities. For example, convolutional neural networks are highly performant on image classification, and sequence models are the state-of-the-art for text generation. However, media such as Internet memes often consist of multiple modalities. A meme may have an image component and a text component, each of which contribute information about what the meme is trying to convey. To extract features from multimodal data, we leverage multimodal deep learning, in which we use multiple feature extractor networks to learn the separate modes individually, and an aggregator network to combine the features to produce the final output classification. We scrape Reddit meme subreddits for post data, including: subreddit name, upvote/downvote count, images, meme text via OCR (or human OCR), and post titles. We construct a train and test set and evaluate results using a precision/accuracy measure for subreddit name predictions. To optimize our model, we use FAIR\u2019s open source multimodal library, Pythia/MMF (",Object(n.jsx)("a",{href:"https://mmf.sh/",rel:"nofollow",children:"https://mmf.sh/"}),"), and try a variety of model architectures and hyperparameters. Finally, we include our best model for demonstration purposes."]})]}),Object(n.jsx)(m.a,{className:e.panel,style:{textAlign:"center"},children:Object(n.jsx)(f.a,{ratio:"16 / 9",style:{maxWidth:"60%",left:"50%",transform:"translate(-50%, 0)"},children:Object(n.jsx)("iframe",{src:"https://www.youtube-nocookie.com/embed/LpU8CUmxcI8",frameBorder:"0",allow:"accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture",allowFullScreen:!0})})}),Object(n.jsxs)(m.a,{className:e.panel,children:[Object(n.jsx)(h.a,{variant:"h4",style:{textAlign:"center"},gutterBottom:!0,children:"Examples"}),Object(n.jsx)(c.a,{children:Object(n.jsx)(w.a,{autoPlay:!1,animation:"slide",indicators:!0,timeout:500,navButtonsAlwaysVisible:!0,navButtonsAlwaysInvisible:!1,children:[Object(n.jsx)(m.a,{className:e.examplecard,children:Object(n.jsx)(c.a,{style:{textAlign:"center",width:"50%"},children:Object(n.jsx)(y.a,{src:T,alt:"Results for language models",aspectRatio:5/3})})}),Object(n.jsx)(m.a,{className:e.examplecard,children:Object(n.jsx)(c.a,{style:{textAlign:"center",width:"50%"},children:Object(n.jsx)(y.a,{src:A,alt:"Results for language models",aspectRatio:5/3})})}),Object(n.jsx)(m.a,{className:e.examplecard,children:Object(n.jsx)(c.a,{style:{textAlign:"center",width:"50%"},children:Object(n.jsx)(y.a,{src:I,alt:"Results for language models",aspectRatio:5/3})})})]})})]}),Object(n.jsxs)(m.a,{className:"".concat(e.panel," ").concat(e.shinypanel),children:[Object(n.jsx)(h.a,{variant:"h4",style:{textAlign:"center"},gutterBottom:!0,children:"Try It Yourself!"}),Object(n.jsx)(c.a,{style:{textAlign:"center"},children:Object(n.jsx)(u.a,{className:e.shinybutton,href:"https://colab.research.google.com/drive/1139WDXzKaWsXPr2rUKzH5Vt8C8ZnFA9k",children:"Check it out on Google Colab"})})]}),Object(n.jsxs)(m.a,{className:e.panel,children:[Object(n.jsx)(h.a,{variant:"h4",style:{textAlign:"center"},gutterBottom:!0,children:"Behind The Scenes"}),Object(n.jsxs)(m.a,{className:e.detailtext,children:[Object(n.jsx)(h.a,{variant:"h5",style:{textAlign:"center"},gutterBottom:!0,children:"Related Work"}),Object(n.jsx)(h.a,{gutterBottom:!0,children:"Our project was inspired by Facebook AI\u2019s \u201cHateful Memes Challenge\u201d in which participants developed novel model architectures to detect harmful multimodal content. The Facebook meme dataset consists of ~10,000 multimodal examples. Based on our personal understanding of the Internet meme culture, we decided that a larger dataset needed to be collected in order to reasonably represent the various subcultures on the Internet. Accordingly, we scraped meme data from Reddit, a popular hub for sharing meme content, which totals 70,000+ examples. Additionally, instead of a binary classification problem, we defined a multi-class classification problem in which our model has to output the most apt Reddit subreddit that a given meme fits. The rationale behind this decision was that different communities on the Internet operate by different de facto rules and guidelines. It is difficult to prescribe a blanket hateful/un-hateful categorization for all memes shared on the Internet. From a human perspective, a meme is usually considered within a given context. For example, particularly dark or edgy jokes may be perfectly acceptable in a community such as r/dankmemes but unacceptable according to Facebook\u2019s platform guidelines. Hence, our work on multimodal multi-class classification could serve as a first step into exploring the effect that the style of multimodal content has on whether or not it is considered hateful. Furthermore, our model can be used by meme creators in deciding where to best post their meme. Optimizing this portion of the meme economy could be highly impactful in facilitating a less hateful Internet community, because by determining the most appropriate channels for memes, creators can avoid posting their work to places where it would be negatively received."})]}),Object(n.jsxs)(m.a,{className:e.detailtext,children:[Object(n.jsx)(h.a,{variant:"h5",style:{textAlign:"center"},gutterBottom:!0,children:"Methodology"}),Object(n.jsx)(h.a,{gutterBottom:!0,children:"The final goal of this project was to build a multimodal model to predict which subreddit a meme was posted in, using both the image and title text components. First, we scraped around 80,000 posts worth of meme data from Reddit (including important metadata such as post title, number of upvotes, etc) and labeled each meme with the subreddit it was sourced from. We included 19 different subreddits, which represents a good variety of multimodal content that people enjoy sharing on the Internet. Importantly, while some subreddits are stylistically very distinct from each other yet feature similar post title conventions (dankmemes vs. tumblr), others are more difficult to distinguish from each other just by looking at the image, even for humans (meirl vs. 2meirl4meirl). Hence, if our model is able to achieve high predictive performance, we can be certain to a large extent that our methodology is appropriately enabling the model to extract information from both text and image modalities."}),Object(n.jsx)(h.a,{gutterBottom:!0,children:"Next, we created some baseline models that classify a meme based on text only, as well as models based on image only. The results of these models were compared to a model that classified based on both image and text, with the expectation that the multimodal model should perform better because it has more information it can use to predict. Additionally, we utilized transfer learning by incorporating BERT into the language portion of our architecture, and using a network pre-trained on ImageNet for the image portion of our architecture. This decision allowed us to leverage well-engineered basic features and focus our development efforts on learning features that are more unique to memes."})]}),Object(n.jsxs)(m.a,{className:e.detailtext,children:[Object(n.jsx)(h.a,{variant:"h5",style:{textAlign:"center"},gutterBottom:!0,children:"Experiments & Results"}),Object(n.jsx)(h.a,{variant:"h6",style:{textAlign:"center"},gutterBottom:!0,children:"Human Branch (Humans)"}),Object(n.jsx)(h.a,{gutterBottom:!0,children:"Before building any models, we established human-level performance by having our team of five label 91 randomly sampled memes from our dataset and calculating our combined accuracy."}),Object(n.jsx)(h.a,{variant:"h6",style:{textAlign:"center"},gutterBottom:!0,children:"Language Branch (BERT)"}),Object(n.jsx)(h.a,{gutterBottom:!0,children:"The next experiment we performed was with the baseline language model. This model used word level representations of the title, instead of character level representations because we thought that key words in the title would be important for predicting what subreddit a meme would be posted in. For this approach, we scanned the 7000 titles of the posts, and for each word that appeared more than 30 times, it was assigned a number. There were 2420 words which were considered part of our vocab in the post titles, so each title was converted to a 1 by 2420 vector, where the index \u201ci\u201d in the vector was set to 1 if the word assigned the number \u201ci\u201d appeared in the title. This vector was used as an input to a linear network, with the first layer having 2420 input neurons and 512 output neurons, while the second layer had 512 input neurons and 20 output neurons because there were 20 output classes to predict. The activation functions used were leaky relu, and the network was trained for 40 epochs with a learning rate of 0.01. Using a subset of the data with 8000 train samples and 2000 test samples, which only had 16 classes (the second linear layer was changed to have 16 output neurons), the model was able to get 51% accuracy on the test set. When using the full dataset with around 70000 examples total, and 20 output classes, the model reached around 33% accuracy after training for 40 epochs, and the accuracy did not appear to increase with further training."}),Object(n.jsx)(h.a,{gutterBottom:!0,children:"Because the full data set of 70,000 post titles (50,000 train) likely doesn\u2019t capture enough language to generalize well, we then decided to try a pretrained model, namely BERT. This was done via the pretrained BERT library in PyTorch. Using the built-in tokenizer in the package, we separated each post title to 20 words (either truncating if there were more or padding with \u201c[PAD]\u201d) then used this as input into the model, which output a 1x20x768 tensor for 1 sample. Besides this we chose not to do any other significant preprocessing to the post titles that would normally be done in other language models, like lemmatizing or removing special characters as the post titles may rely on these features for meaning (eg emojis and utf-8 characters for Subs like r/surrealmemes). We then fed this output into a few fully-connected layers with ReLU activation before the prediction layer. However, due to computational limitations we added a convolutional layer after the output of BERT to downsize the number of weights needed in the fully-connected layers. Splitting the full data set into 80% train and 20% validation, after training for 4 epochs the accuracy was hovering around 0.62869 on the validation set (9028/14360)."}),Object(n.jsx)(c.a,{style:{textAlign:"center",width:"50%"},children:Object(n.jsx)(y.a,{src:B,alt:"Results for language models",aspectRatio:8/3})}),Object(n.jsx)(h.a,{variant:"h6",style:{textAlign:"center"},gutterBottom:!0,children:"Image Branch (VGG-11)"}),Object(n.jsx)(h.a,{gutterBottom:!0,children:"The second experiment we performed focused on the baseline image model. Instead of initializing a convolutional neural network with random weights, we experimented with a variety of pretrained architectures that are well known for their high predictive performance (results shown in Figure 2 below). This design decision made it much easier to load in the dataset, since many of our examples are very large (on the order of 100M pixels) and transfer learning requires less data. However, we still sample from the full dataset. Specifically, the data was split into a training set and a validation set, with 200 examples in the training set for each category, and 100 examples in the validation set for each category. We chose to fine-tune the convnet on the meme data instead of freezing the pretrained weights as a fixed feature extractor. This decision was supported by our exploratory experimentation in which we found that a pretrained ResNet-18 model, when fine-tuned on meme images, achieved 0.015 higher accuracy when compared to when the weights were frozen. This finding makes sense because meme formats are typically not represented in the ImageNet dataset, so some domain shift is necessary. Also informed by our exploratory experimentation, we apply data augmentation in the form of random crops and horizontal flips. The most impactful hyperparameters we found were learning rate and momentum, which we set to 0.001 and 0.9, respectively. Finally, to aid convergence, we decay the learning rate by a factor of 0.1 every 7 epochs. The results of this experiment guided how we developed the image component of our final model, detailed below."}),Object(n.jsx)(c.a,{style:{textAlign:"center",width:"50%"},children:Object(n.jsx)(y.a,{src:k,alt:"Results for image models",aspectRatio:7/3})}),Object(n.jsx)(h.a,{variant:"h6",style:{textAlign:"center"},gutterBottom:!0,children:"Multimodal Branch"}),Object(n.jsx)(c.a,{style:{textAlign:"center",width:"50%"},children:Object(n.jsx)(y.a,{src:O,alt:"Model structure"})}),Object(n.jsx)(h.a,{gutterBottom:!0,children:"Our final round of experimentation focused on a combined multimodal model. We\u2019ve included a figure of this model\u2019s architecture above. Our model has two modalities: a text modality that takes in a post\u2019s string title, and an image modality that takes in the meme you want to post, formatted as an RGB 3x256x256 tensor. The model has two branches, one for each modality, and obtains derived representations of each modality that it then combines into a single tensor. For the text modality, we simply run the title through a pretrained BERT model. We cap title length at 20 words, and truncate titles longer and pad sentences that are shorter. For the image modality, we had a six layer convolutional network with ReLU activation layers between the convolutional layers, batch norm layers every second layer, dropout layers after the fourth and sixth convolutional layers, and a maxpooling layer after the sixth convolutional layer. After obtaining the outputs of each branch, i.e. the output of the BERT model and the output of the CNN, we fused both modalities together by flattening the outputs and concatenating them, which the text modality\u2019s output coming before the image modality\u2019s output. Then, our model finished with a five layer linearly connected network that outputs a 1 dimensional 19 element tensor where the index of the largest element in the tensor corresponds to a subreddit, which the model predicts the meme and title to be posted in. We ran this multimodal model for 10 epochs over the training data, and achieved a final test accuracy of 91.8%."}),Object(n.jsx)(h.a,{gutterBottom:!0,children:"Our final round of experimentation focused on a combined multimodal model. We\u2019ve included a figure of this model\u2019s architecture above. Our model has two modalities: a text modality that takes in a post\u2019s string title, and an image modality that takes in the meme you want to post, formatted as an RGB 3x256x256 tensor. The model has two branches, one for each modality, and obtains derived representations of each modality that it then combines into a single tensor. For the text modality, we simply run the title through a pretrained BERT model. We cap title length at 20 words, and truncate titles longer and pad sentences that are shorter. For the image modality, we had a six layer convolutional network with ReLU activation layers between the convolutional layers, batch norm layers every second layer, dropout layers after the fourth and sixth convolutional layers, and a maxpooling layer after the sixth convolutional layer. After obtaining the outputs of each branch, i.e. the output of the BERT model and the output of the CNN, we fused both modalities together by flattening the outputs and concatenating them, which the text modality\u2019s output coming before the image modality\u2019s output. Then, our model finished with a five layer linearly connected network that outputs a 1 dimensional 19 element tensor where the index of the largest element in the tensor corresponds to a subreddit, which the model predicts the meme and title to be posted in. We ran this multimodal model for 10 epochs over the training data, and achieved a final test accuracy of 91.8%."}),Object(n.jsx)(h.a,{gutterBottom:!0,children:"Overall, our model did quite well. The multimodal modal model far outstripped the best text only and image only models, which had performances 63.8% and 68.1% respectively. One thing we observed was that for predicting some subs, the image tended to be a pretty big clue, like greentext or deepfriedmemes, while for others, the title was the big giveaway, like me_irl. The multimodal models were able to leverage both modalities, while the unimodal models couldn\u2019t. Additionally, there are other more general subs where it was difficult to tell which sub they belonged to based on the meme or the title alone, like dankmemes. For these subs, the multimodal model was able to detect complex patterns and associations between the title and the image that we humans could not. In fact, humans did worse than all the models, with an accuracy of 62.6%. Humans especially struggled on the more general subreddits, where they were often able to narrow it down to 2 or 3 candidate subreddits, but could not correctly figure out which one of them was the correct subreddit."})]})]}),Object(n.jsx)(m.a,{className:"".concat(e.panel," ").concat(e.shinypanel),children:Object(n.jsx)(c.a,{style:{width:"40%"},children:Object(n.jsx)(y.a,{src:i,alt:"Leo DiCaprio numpy meme (credits: Will Chen)",color:"transparent",onClick:function(e){r(i===j?v:j)},style:{height:"100px"}})})})]})}var z=function(e){e&&e instanceof Function&&a.e(3).then(a.bind(null,207)).then((function(t){var a=t.getCLS,n=t.getFID,o=t.getFCP,i=t.getLCP,r=t.getTTFB;a(e),n(e),o(e),i(e),r(e)}))};s.a.render(Object(n.jsx)(i.a.StrictMode,{children:Object(n.jsx)(F,{})}),document.getElementById("root")),z()},34:function(e,t,a){"use strict";(function(e){a.d(t,"c",(function(){return d})),a.d(t,"b",(function(){return h})),a.d(t,"e",(function(){return m})),a.d(t,"d",(function(){return g})),a.d(t,"a",(function(){return y}));var n=a(15),o=a.n(n),i=a(33),r=a(85),s=a.n(r),l=a(68),c=(a(69),a(20),"https://github.com/davidpfahler/react-ml-app/raw/master/src/dogs-resnet18.onnx"),d=function(e){return e.split("_").map((function(e){return e.charAt(0).toUpperCase()+e.slice(1)})).join(" ")},h=function(e){return"https://i.redd.it/vb4uq6nipk251.jpg"},m=function(){return new l.InferenceSession({backendHint:"webgl"})};function u(e){return p.apply(this,arguments)}function p(){return(p=Object(i.a)(o.a.mark((function e(t){return o.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:case"end":return e.stop()}}),e)})))).apply(this,arguments)}function g(e){return f.apply(this,arguments)}function f(){return(f=Object(i.a)(o.a.mark((function e(t){return o.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,t.loadModel(c);case 2:return e.next=4,u(t);case 4:case"end":return e.stop()}}),e)})))).apply(this,arguments)}var b=function(t){return new Promise((function(a,n){e.setTimeout((function(){return a()}),t)}))},w={maxWidth:299,maxHeight:299,cover:!0,crop:!0,canvas:!0,crossOrigin:"Anonymous",orientation:!0},x=function(e){return new Promise((function(t,a){s()(e,(function(e){return t(e)}),w)}))},y=function(){var e=Object(i.a)(o.a.mark((function e(t,a,n){var i,r,s;return o.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:if(a&&a.current){e.next=2;break}return e.abrupt("return");case 2:return e.next=4,x(t);case 4:if("error"!==(i=e.sent).type){e.next=7;break}throw new Error("could not load image");case 7:return(r=a.current.getContext("2d")).drawImage(i,0,0),e.next=11,b(1);case 11:s=r.getImageData(0,0,a.current.width,a.current.height),console.log("in fetchImage,"),console.log(a.current.width+" "+a.current.height),console.log(s),n(s);case 16:case"end":return e.stop()}}),e)})));return function(t,a,n){return e.apply(this,arguments)}}()}).call(this,a(131))}},[[153,1,2]]]);
//# sourceMappingURL=main.bf2e76cb.chunk.js.map