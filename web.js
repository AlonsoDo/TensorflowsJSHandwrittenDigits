var express = require('express');
var path = require('path');
var app = express();
var mnist = require('mnist');
var tf = require('@tensorflow/tfjs');
var async = require('async');
var bodyParser = require('body-parser');

app.use(bodyParser.urlencoded({extended: true}));

// Define the port to run on
app.set('port',4000);

app.use(express.static(path.join(__dirname, 'public')));

// Listen for requests
var server = app.listen(app.get('port'),function(){
    var port = server.address().port;
    console.log('Server running on port '+port);
});

// Recive the image array from the canvas
app.post('/',function(req,res){   
    
    var arrayImg = req.body.imgData;
    var arrayImgInt = [];
    
    for ( var i = 0; i < arrayImg.length; i++){
        arrayImgInt.push(parseInt(arrayImg[i]));
    }
        
    if (endTraining) {
        //We make the prediction of the image and return the class (from 0 to 9)
        var output = model.predict(tf.tensor4d(arrayImgInt, [1, 28, 28, 1]));
        var prediction = Array.from(output.argMax(1).dataSync());
        console.log('Prediction: ' + prediction[0]);
        res.send('Prediction: ' + prediction[0]);
    }else{
        res.send('Training the model');        
    }
    
});

// This loads 8000 images to train and 100 images for test (not need now).
// Then you can use less to training and test to reduce the wait time.
var set = mnist.set(8000, 100);

var trainingSet = set.training;
var testSet = set.test;

var endTraining = false;

var model = tf.sequential();

model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'VarianceScaling'
      }));
    
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
      }));
    
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'VarianceScaling'
      }));
      
      model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
      }));
      
    model.add(tf.layers.flatten());
    
    model.add(tf.layers.dense({
        units: 10,
        kernelInitializer: 'VarianceScaling',
        activation: 'softmax'
      }));    
    
    // Here all the components of the model are united.
    model.compile({
        optimizer: 'sgd',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      });
    
    var input;
    var label;
    var count = -1;
        
    // Maybe this is the hardest part of understanding how it works.
    // We use the function async.whilst () of the library async.js to simulate
    // a synchronic behavior of the fit () method that is in charge of training the model.
    // When you finish running the loop with whilst, the network is ready to test images and make predictions.    
    async.whilst(
        
        //This executes the loop over images of the training set.
        function() { return count < 7990; },
        function(callback) {
            count++;
            
            // We trick the image that is a normal array of 784 elements (floats) and we turn it into a 4D Tensor
            // with which we feed the model introducing the images one by one.
            input = tf.tensor4d(trainingSet[count].input, [1, 28, 28, 1]);
            
            // We introduce the categories that go from 0 to 9 in a 2D Tensor.
            label = tf.tensor2d(trainingSet[count].output, [1, 10]);            
            
            // We train the model.
            model.fit(input, label,{epochs:1});
            
            // To open the browser console, it is usually done with the right button of the mouse and selecting Inspector.
            // I recommend you open it before to see what is happening.
            console.log('Count: ' + count);
            setTimeout(function() {
                callback(null, count);
            },10);            
        },
        function (err, n) {
            
            //When the training is finished, the network is ready to make predictions.
            console.log('Ready For Testing...')            
            endTraining = true;            
            
        }
    ); 