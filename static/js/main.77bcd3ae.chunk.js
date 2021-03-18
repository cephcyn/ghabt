(this["webpackJsonpganna-have-a-bad-time"]=this["webpackJsonpganna-have-a-bad-time"]||[]).push([[0],{104:function(e,t,a){},155:function(e,t,a){"use strict";a.r(t);var r=a(0),n=a.n(r),i=a(10),o=a.n(i),s=(a(104),a(202)),c=a(205),d=a(61),l=a(163),h=a(74),u=(a(105),a(85)),m=a.n(u),p=(a(106),a(51)),g=a.n(p),b=(a(129),a.p+"static/media/previous_work.f3c920e6.png"),w=a.p+"static/media/455_results.3eabc746.PNG",x=a.p+"static/media/bird_pert.31495ae2.PNG",f=a.p+"static/media/ship_ex.0d0cd767.PNG",j=(a(15),a(33),a(24),a(204),a(203),a(206),a(60),a(82),a(81),a(2));Object(h.a)({icon:{width:"50%",height:"50%",color:"grey"}}),a(86),a(196),Object(h.a)({card:{width:"299px",height:"299px",position:"relative",display:"flex",alignItems:"center",justifyContent:"center",marginBottom:10},canvas:{width:"299px",height:"299px",zIndex:0,position:"absolute"},input:{zIndex:9999,position:"absolute"}});a(207),a(197),a(198),a(199),a(200),a(201),Object(h.a)({card:{height:"auto"},item:{paddingTop:10}});var v=a(34);Object(v.e)(),Object(h.a)((function(e){return{root:{padding:"15px 30px"},demoElement:{marginTop:"10px",marginBottom:"10px",width:"100%"},submit:{background:"linear-gradient(45deg, #d08771, #c85b85)",backgroundSize:"200% 200%",border:0,borderRadius:3,boxShadow:"0 3px 5px 2px rgba(255, 105, 135, .1)",color:"white",height:48,padding:"0 30px"},shiny:{background:"linear-gradient(45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab)",backgroundSize:"400% 400%",animation:"$gradient 15s ease infinite",boxShadow:"0 3px 5px 2px rgba(255, 105, 135, .5)"},"@keyframes gradient":{"0%":{"background-position":"0% 50%"},"50%":{"background-position":"100% 50%"},"100%":{"background-position":"0% 50%"}}}}));var y=Object(h.a)((function(e){return{root:{align:"center"},panel:{padding:"15px 30px",marginTop:"20px",marginBottom:"20px"},examplecard:{padding:"10px 30px",position:"relative",left:"50%",transform:"translate(-50%, 0)",maxWidth:"80%"},memetitletext:{fontFamily:"Comic Sans MS, Comic Sans, Comic Neue, cursive"},detailtext:{padding:"10px 30px",marginTop:"15px",marginBottom:"15px"},shinybutton:{background:"linear-gradient(45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab)",backgroundSize:"400% 400%",animation:"$gradient 15s ease infinite",border:0,borderRadius:3,boxShadow:"0 3px 5px 2px rgba(255, 105, 135, .5)",color:"white",height:48,padding:"0 30px"},shinypanel:{background:"linear-gradient(45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab)",backgroundSize:"400% 400%",animation:"$gradient 15s ease infinite"},"@keyframes gradient":{"0%":{"background-position":"0% 50%"},"50%":{"background-position":"100% 50%"},"100%":{"background-position":"0% 50%"}}}}));function O(){var e=y();return Object(j.jsxs)(s.a,{className:e.root,children:[Object(j.jsx)(c.a,{}),Object(j.jsxs)(l.a,{className:"".concat(e.panel," ").concat(e.shinypanel),style:{textAlign:"center"},children:[Object(j.jsx)(d.a,{variant:"h1",children:"Bullying Models with Perturbation"}),Object(j.jsx)(d.a,{variant:"h4",children:"Which model architectures stand up the best to adversarial image perturbation."})]}),Object(j.jsxs)(l.a,{className:e.panel,children:[Object(j.jsx)(d.a,{variant:"h4",style:{textAlign:"center"},gutterBottom:!0,children:"Problem Description"}),Object(j.jsxs)(d.a,{gutterBottom:!0,children:["In recent decades, neural networks have had a large role in the area of computer vision. Computer vision has recently garnered increasing international attention as it has been applied to the world of surveillance. As a reaction to this, individuals with privacy concerns have increasingly become more interested in how to trick these networks into labeling images incorrectly. In particular, the most popular attacks involve changing what the model would see in small enough ways to trick the model, but still be identifiable to humans. ",Object(j.jsx)("b",{children:"We tested how resistant various neural network architectures would be to a whitebox adversarial attack via image perturbation. We also tested how well using one round of image perturbation as a data augmentation method works to improve model performance"}),"."]}),Object(j.jsx)(d.a,{gutterBottom:!0,children:"Our approach was to investigate a variety of neural network architectures ranging from linear models to the original ResNet model. One of the most common benchmark datasets is the CIFAR-10 dataset, which we used to train and identify the most accurate neural network architectures out of a few broad categories. From here, we devised a way to perturb images via a whitebox adversarial attack on the trained models. This attack consists of taking a model trained on CIFAR-10, and learning how to perturb images by going against the gradient. We then output perturbed images, and then compare how well these models are able to correctly label the images. We found that more complex models tended to perform better on the CIFAR data set, and that as the models increased in complexity, they became increasingly susceptible to our adversarial attack but were better able to learn from it if we trained them using data perturbed in a similar way."})]}),Object(j.jsx)(l.a,{className:e.panel,style:{textAlign:"center"},children:Object(j.jsx)(m.a,{ratio:"16 / 9",style:{maxWidth:"60%",left:"50%",transform:"translate(-50%, 0)"},children:Object(j.jsx)("iframe",{src:"https://www.youtube-nocookie.com/embed/vdVnnMOTe3Q",frameBorder:"0",allow:"accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture",allowFullScreen:!0})})}),Object(j.jsxs)(l.a,{className:e.panel,children:[Object(j.jsx)(d.a,{variant:"h4",style:{textAlign:"center"},gutterBottom:!0,children:"Previous Work"}),Object(j.jsxs)(d.a,{gutterBottom:!0,children:["Our original project proposal was inspired by ",Object(j.jsx)("a",{href:"https://arxiv.org/abs/2002.02196",children:"AI-GAN: Attack-Inspired Generation of Adversarial Examples (Bai et al, 2020)"}),". We wanted to create a GAN that would create perturbed adversarial images in response to a CIFAR-10 trained neural network. However, we found that this would require much more work than originally planned (see further discussion below), so our project shifted away from using GANs to instead generating perturbed images with a whitebox attack."]}),Object(j.jsx)(s.a,{style:{textAlign:"center",width:"50%"},children:Object(j.jsx)(g.a,{src:b,alt:"previous work table",aspectRatio:3})}),Object(j.jsxs)(d.a,{gutterBottom:!0,children:["Another relevant work is ",Object(j.jsx)("a",{href:"https://arxiv.org/pdf/1801.00553.pdf",children:"Threat of Adversarial Attacks on Deep Learning in Computer Vision: A Survey (Akhtar and Mian, IEEE Access 2018)]"})," which examined different methods of performing adversarial attacks on deep learning models with a wide range of approaches (image above related). To contrast with this, in our experiment, we use a uniform attack style but primarily vary the architecture of the model being attacked."]}),Object(j.jsxs)(d.a,{gutterBottom:!0,children:["For our experiment itself, we used the ",Object(j.jsx)("a",{href:"https://www.cs.toronto.edu/~kriz/cifar.html",children:"pre-existing CIFAR-10 dataset"})," and a ",Object(j.jsx)("a",{href:"https://pytorch.org/vision/stable/models.html",children:"pretrained ResNet"})," model."]})]}),Object(j.jsxs)(l.a,{className:e.panel,children:[Object(j.jsx)(d.a,{variant:"h4",style:{textAlign:"center"},gutterBottom:!0,children:"Behind The Scenes"}),Object(j.jsxs)(l.a,{className:e.detailtext,children:[Object(j.jsx)(d.a,{variant:"h5",style:{textAlign:"center"},gutterBottom:!0,children:"Problem Setup"}),Object(j.jsx)(d.a,{gutterBottom:!0,children:"Our first step was to identify and build models. We decided to try four different model architecture types: a linear model, a convolutional model, a convolutional model with batch norm/dropout, and finally a high performance model called ResNet. These architectures were investigated, trained with the CIFAR dataset, and tweaked to improve their accuracy. The most accurate version of each of these architectures was then selected to be used in our attack experiment."}),Object(j.jsx)(d.a,{gutterBottom:!0,children:"Once we had our trained models, we used them to perturb images by calculating the gradient at the input level and adding it to an input perturbation vector. The input-level gradient allows us to note which features would lead the model to label images the way they did, so we use this to ideally perturb images so that these important features wouldn't show up. Finally, we saved the perturbed images generated from each model to create both a perturbed training and testing dataset."}),Object(j.jsx)(d.a,{gutterBottom:!0,children:"After generating our perturbed image datasets, we tested them on our architectures. For each architecture, we trained models in two different ways and compared each trained model\u2019s performance on the original and perturbed image datasets. The first way we trained each model type was with 50 epochs on the original CIFAR training data. The second way we trained each model type was with 25 epochs on the combined (original CIFAR and perturbed) training data. These variants were tested on the perturbed testing data and the original testing data. Finally, we compared the difference in accuracy of these two models across architecture groups."})]}),Object(j.jsxs)(l.a,{className:e.detailtext,children:[Object(j.jsx)(d.a,{variant:"h5",style:{textAlign:"center"},gutterBottom:!0,children:"Data Used"}),Object(j.jsx)(d.a,{gutterBottom:!0,children:"The main dataset that we used was CIFAR-10."}),Object(j.jsx)(d.a,{gutterBottom:!0,children:"The other datasets we used were those of perturbed images generated during the experiment, which were derived from CIFAR-10."})]}),Object(j.jsxs)(l.a,{className:e.detailtext,children:[Object(j.jsx)(d.a,{variant:"h5",style:{textAlign:"center"},gutterBottom:!0,children:"Techniques Used"}),Object(j.jsxs)(d.a,{gutterBottom:!0,children:["We used PyTorch for constructing our models. We also tuned our ResNet model from the pretrained model PyTorch provides. Information on the ResNet model can be found ",Object(j.jsx)("a",{href:"https://pytorch.org/vision/stable/models.html",children:"here"}),"."]}),Object(j.jsx)(d.a,{gutterBottom:!0,children:"Other than those, our model architecture and perturbation code was written ourselves."})]}),Object(j.jsxs)(l.a,{className:e.detailtext,children:[Object(j.jsx)(d.a,{variant:"h5",style:{textAlign:"center"},gutterBottom:!0,children:"Experiments and Results"}),Object(j.jsx)(s.a,{style:{textAlign:"center",width:"50%"},children:Object(j.jsx)(g.a,{src:w,alt:"expereiment results",aspectRatio:2})}),Object(j.jsx)(d.a,{gutterBottom:!0,children:"Overall, we found that more complex models tended to perform better on the CIFAR data set, as expected. One thing that was surprising about this was that as the models increased in complexity, they became increasingly susceptible to our adversarial attack. Interestingly, while the more complex models were less resilient against image perturbation attacks, they did learn to handle perturbed data more adeptly when we trained them on an augmented dataset. For all model architectures we tested, augmenting the input dataset with the perturbed images decreased the final model performance on the original CIFAR test set, but increased performance on the perturbed test set. This could be because the perturbations serve a kind of \u201cspecialized\u201d data augmentation to help the model generalize better to images with a small amount of noise. In this sense, training on this augmented data set for a longer period of time may lead to better overall performance in the long run."}),Object(j.jsx)(s.a,{style:{textAlign:"center",width:"50%"},children:Object(j.jsx)(g.a,{src:x,alt:"example of a perturbed bird image",aspectRatio:1.5})}),Object(j.jsx)(s.a,{style:{textAlign:"center",width:"50%"},children:Object(j.jsx)(g.a,{src:f,alt:"example of a perturbed ship image",aspectRatio:2})})]})]}),Object(j.jsxs)(l.a,{className:e.panel,children:[Object(j.jsx)(d.a,{variant:"h4",style:{textAlign:"center"},gutterBottom:!0,children:"Discussion"}),Object(j.jsxs)(l.a,{className:e.detailtext,children:[Object(j.jsx)(d.a,{variant:"h5",style:{textAlign:"center"},gutterBottom:!0,children:"Problems Encountered"}),Object(j.jsx)(d.a,{gutterBottom:!0,children:"The first problem that we encountered encouraged a shift in our overall project approach. We had originally planned on using a GAN to generate adversarial perturbed images to attack our model, but when we tried this, we noticed how the GAN would create adversarial datasets containing images that don't match their label. Resolving this issue would involve a much larger code base than we believed we could develop, so we had to shift our project to a whitebox adversarial attack without the use of GAN's."}),Object(j.jsx)(d.a,{gutterBottom:!0,children:"Other than this, most problems we encountered were small bugs or difficulties with PyTorch and Google Colab. In particular, we repeatedly had issues with the models not training at all or having frustratingly long runtimes for training."})]}),Object(j.jsxs)(l.a,{className:e.detailtext,children:[Object(j.jsx)(d.a,{variant:"h5",style:{textAlign:"center"},gutterBottom:!0,children:"Next Steps"}),Object(j.jsx)(d.a,{gutterBottom:!0,children:"If we had more computational resources, we would explore more neural network architectures and see how those are affected by our adversarial attack. Alternatively, we could build on our current analysis by using visualizations to try to understand what accounts for the difference in impacts of image perturbation."}),Object(j.jsx)(d.a,{gutterBottom:!0,children:"We would also be interested in further exploring the idea of using GAN's and performing a black box adversarial attack."}),Object(j.jsx)(d.a,{gutterBottom:!0,children:"Finally, we would be interested in evaluating how well different neural network architectures respond to a more closed-loop GAN-based model training process."})]}),Object(j.jsxs)(l.a,{className:e.detailtext,children:[Object(j.jsx)(d.a,{variant:"h5",style:{textAlign:"center"},gutterBottom:!0,children:"How our approach differs from others"}),Object(j.jsx)(d.a,{gutterBottom:!0,children:"Our approach was a study of the various components that exist in modern computer vision deep neural networks and how they are affected by imager perturbations."}),Object(j.jsx)(d.a,{gutterBottom:!0,children:"Compared to other studies, we examined the architecture of the model being attacked instead of the architecture of the attack or method of generating adversarial examples itself. This means we address an aspect of adversarial input attacks that other surveys have not examined in detail, which is overall beneficial. However, as mentioned in our discussion of future steps, it would be very interesting to do a multi-dimensional analysis studying interaction of different attack methods and attacked model architectures together as well."})]})]})]})}var k=function(e){e&&e instanceof Function&&a.e(3).then(a.bind(null,209)).then((function(t){var a=t.getCLS,r=t.getFID,n=t.getFCP,i=t.getLCP,o=t.getTTFB;a(e),r(e),n(e),i(e),o(e)}))};o.a.render(Object(j.jsx)(n.a.StrictMode,{children:Object(j.jsx)(O,{})}),document.getElementById("root")),k()},34:function(e,t,a){"use strict";(function(e){a.d(t,"c",(function(){return l})),a.d(t,"b",(function(){return h})),a.d(t,"e",(function(){return u})),a.d(t,"d",(function(){return g})),a.d(t,"a",(function(){return j}));var r=a(15),n=a.n(r),i=a(33),o=a(84),s=a.n(o),c=a(68),d=(a(69),a(19),"https://github.com/davidpfahler/react-ml-app/raw/master/src/dogs-resnet18.onnx"),l=function(e){return e.split("_").map((function(e){return e.charAt(0).toUpperCase()+e.slice(1)})).join(" ")},h=function(e){return"https://i.redd.it/vb4uq6nipk251.jpg"},u=function(){return new c.InferenceSession({backendHint:"webgl"})};function m(e){return p.apply(this,arguments)}function p(){return(p=Object(i.a)(n.a.mark((function e(t){return n.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:case"end":return e.stop()}}),e)})))).apply(this,arguments)}function g(e){return b.apply(this,arguments)}function b(){return(b=Object(i.a)(n.a.mark((function e(t){return n.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,t.loadModel(d);case 2:return e.next=4,m(t);case 4:case"end":return e.stop()}}),e)})))).apply(this,arguments)}var w=function(t){return new Promise((function(a,r){e.setTimeout((function(){return a()}),t)}))},x={maxWidth:299,maxHeight:299,cover:!0,crop:!0,canvas:!0,crossOrigin:"Anonymous",orientation:!0},f=function(e){return new Promise((function(t,a){s()(e,(function(e){return t(e)}),x)}))},j=function(){var e=Object(i.a)(n.a.mark((function e(t,a,r){var i,o,s;return n.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:if(a&&a.current){e.next=2;break}return e.abrupt("return");case 2:return e.next=4,f(t);case 4:if("error"!==(i=e.sent).type){e.next=7;break}throw new Error("could not load image");case 7:return(o=a.current.getContext("2d")).drawImage(i,0,0),e.next=11,w(1);case 11:s=o.getImageData(0,0,a.current.width,a.current.height),console.log("in fetchImage,"),console.log(a.current.width+" "+a.current.height),console.log(s),r(s);case 16:case"end":return e.stop()}}),e)})));return function(t,a,r){return e.apply(this,arguments)}}()}).call(this,a(132))}},[[155,1,2]]]);
//# sourceMappingURL=main.77bcd3ae.chunk.js.map