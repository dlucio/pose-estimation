// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
PoseNet example using p5.js
Modified by Dj Soul (dlucio@impa.br)
=== */

let video;
let videoROI;
let poseNet;
let stats;
let poses = [];

// Region Of Interest of the video that will be copied to videoROI
let ROI = {
  sx: 0, sy: 0, sw: 100, sh: 100, // src x,y,w,h
  dx: 0, dy: 0, dw: 100, dh: 100, // dst x,y,w,h
};
let roiControllers = {
  x: null,
  y: null,
  w: null,
  h: null
}

const videoSrc = { 
  'video 1': ['../assets/u2_640x360.mp4'], 
  'video 2': ['../assets/frevo_640x360.mp4'],
  'video 3': ['../assets/pomplamoose_640x360.mp4'],
  'video 4': ['../assets/dancing_640x360.mp4'],
  'video 5': ['../assets/soccer_video_640x360.mp4'],
}

let isModelReady = false;
let isGUIReady = false;

let minPoseConfidence;
let minPartConfidence;

const defaultQuantBytes = 2;

const defaultMobileNetMultiplier = 0.75; //isMobile() ? 0.50 : 0.75;
const defaultMobileNetStride = 16;
const defaultMobileNetInputResolution = 257;

const defaultResNetMultiplier = 1.0;
const defaultResNetStride = 32;
const defaultResNetInputResolution = 257;

const config = {
  architecture: 'MobileNetV1',
  imageScaleFactor: 0.3,
  outputStride: defaultMobileNetStride,
  flipHorizontal: false,
  minConfidence: 0.625,
  maxPoseDetections: 10,
  scoreThreshold: 0.65,
  nmsRadius: 20,
  detectionType: 'multiple',
  inputResolution: defaultMobileNetInputResolution, 
  multiplier: defaultMobileNetMultiplier,
  quantBytes: 2
}

function setup() {

  video = createVideo(videoSrc['video 1'], () => {
    video.loop();
    video.volume(0);
    video.pause();
    video.showControls();
    const w = video.width;
    const h = video.height;
    let canvas = createCanvas(w, h);
    canvas.parent('sketch-holder');

    videoROI = createImage(w, h);
    videoROI.loadPixels();
    // Region Of Interest of the video that will be copied to videoROI
    ROI = {
      sx: 0, sy: 0, sw: width, sh: height, // src x,y,w,h
      dx: 0, dy: 0, dw: width, dh: height, // dst x,y,w,h
    };
    setupGui();
  });
  video.parent( 'video-holder' );

  setupPoseNet();

  stats = new Stats();
  stats.showPanel( 0 ); // 0: fps, 1: ms, 2: mb, 3+: custom
  document.body.appendChild( stats.dom );

}

let centroidTracker = null;
function setupPoseNet() {
  isModelReady = false;  
  
  if (centroidTracker != null) {
    centroidTracker.dispose();
    delete centroidTracker;
  }

  poses = [];

  if (poseNet) {
    poseNet.net.dispose();
    poseNet = null;
  }
  // Create a new poseNet method with a single detection
  // poseNet = ml5.poseNet(video, config,  modelReady);
  poseNet = ml5.poseNet(config,  modelReady);

  // This sets up an event that fills the global variable "poses"
  // with an array every time new poses are detected
  poseNet.on('pose', function (results) {
    poses = [];
    results.forEach( (person, id) => {
      // console.log('score & confidence', person.pose.score, minPoseConfidence);
      person.id = id;
      
      if (person.pose.score >= minPoseConfidence) {
        poses.push(person);
      }
    });
    
  });
}

function modelReady() {
  select('#status').html('Model Loaded');
  select('#status').class('hidden');

  isModelReady = true;
  video.play();
  centroidTracker = new CentroidTracker(180);
}


const tryResNetButtonName = 'tryResNetButton';
const tryResNetButtonText = '[New] Try ResNet50';

const guiState = {
  algorithm: 'multi-pose',
  source: 'video 1', // video n or webcam
  input: {
    architecture: 'MobileNetV1',
    outputStride: defaultMobileNetStride,
    inputResolution: defaultMobileNetInputResolution,
    multiplier: defaultMobileNetMultiplier,
    quantBytes: defaultQuantBytes
  },
  singlePoseDetection: {
    minPoseConfidence: 0.1,
    minPartConfidence: 0.5,
  },
  multiPoseDetection: {
    maxPoseDetections: config.maxPoseDetections,
    minPoseConfidence: config.minConfidence,
    minPartConfidence: config.scoreThreshold,
    nmsRadius: config.nmsRadius,
  },
  output: {
    videoOpacity: 1.0,
    showVideo: true,
    showSkeleton: false,
    showPoints: false,
    showBoundingBox: true,
  },
  estimatePoseEnable: true,
  trackingEnable: true,
};

/**
 * Sets up dat.gui controller on the top-right of the window
 */
function setupGui() {
  const gui = new dat.GUI({width: 300});

  let architectureController = null;
  guiState[tryResNetButtonName] = function() {
    architectureController.setValue('ResNet50')
  };
  // gui.add(guiState, tryResNetButtonName).name(tryResNetButtonText);
  gui.add(guiState, 'estimatePoseEnable')
    .name('Estimate Pose')
    .onChange( (value) => {
      poseNet.estimatePoseEnable = value;
      if (value) {
        poseNet.load().then( () => console.log("Model Reloaded after Estimate Enable.") );
      }
    });  

  gui.add(guiState, 'source', [...Object.keys(videoSrc), 'webcam']).name('Source')
    .onChange( (value) => {
      video.stop();
      video.remove();
      video = null;
      clear();
      
      if (value !== 'webcam') {
        
        video = createVideo(videoSrc[value], () => {
          video.loop();
          video.volume(0);
          video.showControls();
          const w = video.width;
          const h = video.height;
          resizeCanvas(w, h);
          
          ROI = {
            sx: 0, sy: 0, sw: width, sh: height, // src x,y,w,h
            dx: 0, dy: 0, dw: width, dh: height, // dst x,y,w,h
          };
          videoROI = null;
          videoROI = createImage(width, height);
          videoROI.loadPixels();
          roiControllers.x.max(width);  roiControllers.x.setValue(0);      roiControllers.x.updateDisplay();
          roiControllers.y.max(height); roiControllers.y.setValue(0);      roiControllers.y.updateDisplay();
          roiControllers.w.max(width);  roiControllers.w.setValue(width);  roiControllers.w.updateDisplay();
          roiControllers.h.max(height); roiControllers.h.setValue(height); roiControllers.h.updateDisplay();
        });
        config.flipHorizontal = false;

      } else {

        video = createCapture(VIDEO, () => {
          const w = video.width;
          const h = video.height;
          console.log('Why WxH =', w,h);
          
          // NOTE: hardcode canvas size for my camera
          // I don't know why video withxheight == 300x150;
          resizeCanvas(640, 480);
          video.style("transform", "scaleX(-1)");

          ROI = {
            sx: 0, sy: 0, sw: width, sh: height, // src x,y,w,h
            dx: 0, dy: 0, dw: width, dh: height, // dst x,y,w,h
          };
          videoROI = null;
          videoROI = createImage(width, height);
          videoROI.loadPixels();
          roiControllers.x.max(width);  roiControllers.x.setValue(0);      roiControllers.x.updateDisplay();
          roiControllers.y.max(height); roiControllers.y.setValue(0);      roiControllers.y.updateDisplay();
          roiControllers.w.max(width);  roiControllers.w.setValue(width);  roiControllers.w.updateDisplay();
          roiControllers.h.max(height); roiControllers.h.setValue(height); roiControllers.h.updateDisplay();
          
        });
        config.flipHorizontal = true;
        
      }
      video.parent( 'video-holder' );
      
      setupPoseNet();

    });
  // The single-pose algorithm is faster and simpler but requires only one
  // person to be in the frame or results will be innaccurate. Multi-pose works
  // for more than 1 person
  const algorithmController =
      gui.add(guiState, 'algorithm', ['single-pose', 'multi-pose']);

  // The input parameters have the most effect on accuracy and speed of the
  // network
  let input = gui.addFolder('Input');
  // Architecture: there are a few PoseNet models varying in size and
  // accuracy. 1.01 is the largest, but will be the slowest. 0.50 is the
  // fastest, but least accurate.
  architectureController =
      input.add(guiState.input, 'architecture', ['MobileNetV1', 'ResNet50']);
  guiState.architecture = guiState.input.architecture;
  // Input resolution:  Internally, this parameter affects the height and width
  // of the layers in the neural network. The higher the value of the input
  // resolution the better the accuracy but slower the speed.
  let inputResolutionController = null;
  function updateGuiInputResolution(
      inputResolution,
      inputResolutionArray,
  ) {
    if (inputResolutionController) {
      inputResolutionController.remove();
    }
    guiState.inputResolution = inputResolution;
    guiState.input.inputResolution = inputResolution;
    inputResolutionController =
        input.add(guiState.input, 'inputResolution', inputResolutionArray);
    inputResolutionController.onChange(function(inputResolution) {
      guiState.changeToInputResolution = inputResolution;
    });
  }

  // Output stride:  Internally, this parameter affects the height and width of
  // the layers in the neural network. The lower the value of the output stride
  // the higher the accuracy but slower the speed, the higher the value the
  // faster the speed but lower the accuracy.
  let outputStrideController = null;
  function updateGuiOutputStride(outputStride, outputStrideArray) {
    if (outputStrideController) {
      outputStrideController.remove();
    }
    guiState.outputStride = outputStride;
    guiState.input.outputStride = outputStride;
    outputStrideController =
        input.add(guiState.input, 'outputStride', outputStrideArray);
    outputStrideController.onChange(function(outputStride) {
      guiState.changeToOutputStride = outputStride;
    });
  }

  // Multiplier: this parameter affects the number of feature map channels in
  // the MobileNet. The higher the value, the higher the accuracy but slower the
  // speed, the lower the value the faster the speed but lower the accuracy.
  let multiplierController = null;
  function updateGuiMultiplier(multiplier, multiplierArray) {
    if (multiplierController) {
      multiplierController.remove();
    }
    guiState.multiplier = multiplier;
    guiState.input.multiplier = multiplier;
    multiplierController =
        input.add(guiState.input, 'multiplier', multiplierArray);
    multiplierController.onChange(function(multiplier) {
      guiState.changeToMultiplier = multiplier;
    });
  }

  // QuantBytes: this parameter affects weight quantization in the ResNet50
  // model. The available options are 1 byte, 2 bytes, and 4 bytes. The higher
  // the value, the larger the model size and thus the longer the loading time,
  // the lower the value, the shorter the loading time but lower the accuracy.
  let quantBytesController = null;
  function updateGuiQuantBytes(quantBytes, quantBytesArray) {
    if (quantBytesController) {
      quantBytesController.remove();
    }
    guiState.quantBytes = +quantBytes;
    guiState.input.quantBytes = +quantBytes;
    quantBytesController =
        input.add(guiState.input, 'quantBytes', quantBytesArray);
    quantBytesController.onChange(function(quantBytes) {
      guiState.changeToQuantBytes = +quantBytes;
    });
  }

  function updateGui() {
    if (guiState.input.architecture === 'MobileNetV1') {
      updateGuiInputResolution(
          defaultMobileNetInputResolution, [257, 353, 449, 513, 801]);
      updateGuiOutputStride(defaultMobileNetStride, [8, 16]);
      updateGuiMultiplier(defaultMobileNetMultiplier, [0.50, 0.75, 1.0])
    } else {  // guiState.input.architecture === "ResNet50"
      updateGuiInputResolution(
          defaultResNetInputResolution, [257, 353, 449, 513, 801]);
      updateGuiOutputStride(defaultResNetStride, [32, 16]);
      updateGuiMultiplier(defaultResNetMultiplier, [1.0]);
    }
    updateGuiQuantBytes(defaultQuantBytes, [1, 2, 4]);
  }

  updateGui();
  input.open();
  // Pose confidence: the overall confidence in the estimation of a person's
  // pose (i.e. a person detected in a frame)
  // Min part confidence: the confidence that a particular estimated keypoint
  // position is accurate (i.e. the elbow's position)
  let single = gui.addFolder('Single Pose Detection');
  single.add(guiState.singlePoseDetection, 'minPoseConfidence', 0.0, 1.0)
        .onChange( (value) => poseNet.minConfidence = value );
  single.add(guiState.singlePoseDetection, 'minPartConfidence', 0.0, 1.0)
        .onChange( (value) => poseNet.scoreThreshold = value );

  let multi = gui.addFolder('Multi Pose Detection');
  multi.add(guiState.multiPoseDetection, 'maxPoseDetections')
      .min(1)
      .max(20)
      .step(1)
      .onChange( (value) => poseNet.maxPoseDetections = value );
  multi.add(guiState.multiPoseDetection, 'minPoseConfidence', 0.0, 1.0)
      .onChange( (value) => poseNet.minConfidence = value );
  multi.add(guiState.multiPoseDetection, 'minPartConfidence', 0.0, 1.0)
      .onChange( (value) => poseNet.scoreThreshold = value );
  // nms Radius: controls the minimum distance between poses that are returned
  // defaults to 20, which is probably fine for most use cases
  multi.add(guiState.multiPoseDetection, 'nmsRadius').min(0.0).max(40.0)
      .onChange( (value) => poseNet.nmsRadius = value );
  multi.open();

  let output = gui.addFolder('Output');
  output.add(guiState.output, 'videoOpacity', 0.0, 1.0);
  // output.add(guiState.output, 'showVideo');
  output.add(guiState.output, 'showSkeleton');
  output.add(guiState.output, 'showPoints');
  output.add(guiState.output, 'showBoundingBox');
  output.open();


  architectureController.onChange(function(architecture) {
    // if architecture is ResNet50, then show ResNet50 options
    updateGui();
    guiState.changeToArchitecture = architecture;
  });

  algorithmController.onChange(function(value) {
    switch (guiState.algorithm) {
      case 'single-pose':
        multi.close();
        single.open();
        poseNet.detectionType = 'single';
        break;
      case 'multi-pose':
        single.close();
        multi.open();
        poseNet.detectionType = 'multiple';
        break;
    }
  });

  console.log('resolution', width, height);
    
  let roiFolder = gui.addFolder('ROI');
  roiFolder.open();
  roiControllers.x = roiFolder.add(ROI, 'sx')
    .name('x')
    .min(0)
    .max(width)
    .onChange( (value) => { ROI.sx = ROI.dx = value; } );
    roiControllers.y = roiFolder.add(ROI, 'sy')
    .name('y')
    .min(0)
    .max(height)
    .onChange( (value) => { ROI.sy = ROI.dy = value; } );
    roiControllers.w = roiFolder.add(ROI, 'sw')
    .name('w')
    .min(0)
    .max(width)
    .onChange( (value) => { ROI.sw = ROI.dw = value; } );
    roiControllers.h = roiFolder.add(ROI, 'sh')
    .name('h')
    .min(0)
    .max(height)
    .onChange( (value) => { ROI.sh = ROI.dh = value; } );

  console.log(roiControllers.x);
  
  // OpenCV options - prepared to lucas-kanade
  class Parameters {
    constructor() {
      this.useAllKeypoints = false;
      this.showBoundingBox = false;
      this.showCentroid = false;
    }
  };

  trackingParams = new Parameters();
  let trackingFolder = gui.addFolder('Centroid Tracking Algorithm');
  trackingFolder.open();
  trackingFolder.add(guiState, 'trackingEnable').name('Tracking');
  trackingFolder.add(trackingParams, 'useAllKeypoints');
  trackingFolder.add(trackingParams, 'showCentroid');
  trackingFolder.add(trackingParams, 'showBoundingBox');

  isGUIReady = true;

}

let trackingParams;



async function poseDetectionFrame() {
  if (guiState.changeToArchitecture) {
    poseNet.architecture = guiState.changeToArchitecture;
    poseNet.load().then( () => console.log("Model Reloaded after architecture changed.") );

    guiState.architecture = guiState.changeToArchitecture;
    guiState.changeToArchitecture = null;

    poses = [];
  }

  if (guiState.changeToMultiplier) {
    poseNet.multiplier = +guiState.changeToMultiplier;
    poseNet.load().then( () => console.log("Model Reloaded after multiplier changed.") );
    
    guiState.multiplier = +guiState.changeToMultiplier;
    guiState.changeToMultiplier = null;

    poses = [];
  }

  if (guiState.changeToOutputStride) {
    poseNet.outputStride = +guiState.changeToOutputStride;
    poseNet.load().then( () => console.log("Model Reloaded after outputStride changed.") );
    
    guiState.outputStride = +guiState.changeToOutputStride;
    guiState.changeToOutputStride = null;

    poses = [];
  }

  if (guiState.changeToInputResolution) {
    poseNet.inputResolution = +guiState.changeToInputResolution;
    poseNet.load().then( () => console.log("Model Reloaded after inputResolution changed.") );
    
    guiState.inputResolution = +guiState.changeToInputResolution;
    guiState.changeToInputResolution = null;

    poses = [];
  }

  if (guiState.changeToQuantBytes) {
    poseNet.quantBytes = guiState.changeToQuantBytes;
    poseNet.load().then( () => console.log("Model Reloaded after quantBytes changed.") );

    guiState.quantBytes = guiState.changeToQuantBytes;
    guiState.changeToQuantBytes = null;

    poses = [];
  }

  if (poses.length == 0) {
    // reset Centroid Tracker
    centroidTracker.dispose();
  }

  switch (guiState.algorithm) {
    case 'single-pose':
      minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
      minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;
      poseNet.singlePose(videoROI);
      break;
    case 'multi-pose':
      minPoseConfidence = +guiState.multiPoseDetection.minPoseConfidence;
      minPartConfidence = +guiState.multiPoseDetection.minPartConfidence;
      poseNet.multiPose(videoROI);
      break;
  }
}

function draw() {
  
  stats.begin();
  
  clear();
  video.style('opacity', guiState.output.videoOpacity);

  if (typeof(videoROI) !== 'undefined') {
    videoROI.updatePixels();
    videoROI.copy(video, ...Object.values(ROI));

    if (guiState.source === 'webcam') {
      scale(-1);
    }
    image(videoROI,0,0);
    if (guiState.source === 'webcam') {
      scale(-1);
    }

    noFill();
    strokeWeight(4);
    stroke(255, 255, 25);
    rect(ROI.sx, ROI.sy, ROI.sw, ROI.sh);
  }

  if (guiState.estimatePoseEnable && isModelReady && isGUIReady) {
    
    poseDetectionFrame().then( () => {
      

      // For each person´s pose detected in an image, loop through the poses
      // and draw the resulting skeleton and keypoints if over certain confidence
      // scores
      poses.forEach( (person, index) => {
        const pose = person.pose;

        if (pose.score >= minPoseConfidence) {
          if (guiState.output.showPoints) {
            drawKeypoints(pose.keypoints);
          }
          if (guiState.output.showSkeleton) {
            drawSkeleton(person.skeleton);
          }
          if (guiState.output.showBoundingBox) {
            drawBoundingBox(person.boundingBox, person.id, index)
          }
        }
        
      });
        
    });

    if (guiState.trackingEnable && isModelReady && isGUIReady && centroidTracker != null) {

      const showUntil = 5;
      const ob = centroidTracker.update(poses, trackingParams.useAllKeypoints);
      if (typeof(ob) !== 'undefined') {
        const objectsIDs = Object.keys(ob.objects);
        objectsIDs.forEach(oid => {
          const c = ob.objects[oid];
          const x = c[0];
          const y = c[1];
  
          if (centroidTracker.disappeared[oid] < showUntil) {
  
            fill(0, 255, 255);
            stroke(255, 0, 0);
            strokeWeight(1);
            if (trackingParams.showCentroid) {
              ellipse(x, y, 10);
            }
    
            textSize(20);
            text(`Person ${oid}`, x+10, y-10);
            
          }
  
        });
  
        if (trackingParams.showBoundingBox) {
          const bboxesIDs = Object.keys(ob.bboxes);
          bboxesIDs.forEach(bid => {
            const bb = ob.bboxes[bid];
  
            if (centroidTracker.disappeared[bid] < showUntil) {
              noFill();
              stroke(0, 255, 0);
              strokeWeight(2);
              rect(bb.x0, bb.y0, bb.x1-bb.x0, bb.y1-bb.y0);
            }
          });
        }
      }
    }    
  }
  stats.end();
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints(keypoints) {
  for (let j = 0; j < keypoints.length; j++) {
    // A keypoint is an object describing a body part (like rightArm or leftShoulder)
    let keypoint = keypoints[j];
    // Only draw an ellipse is the pose probability is bigger than minPartConfidence
    if (keypoint.score > minPartConfidence) {
      fill(255, 0, 0);
      noStroke();
      const r=9;
      ellipse(keypoint.position.x, keypoint.position.y, r, r);
    }
  }
}

// A function to draw the skeletons
function drawSkeleton(skeleton) {
  // For every skeleton, loop through all body connections
  for (let j = 0; j < skeleton.length; j++) {
    let partA = skeleton[j][0];
    let partB = skeleton[j][1];
    stroke(255, 0, 0);
    strokeWeight(3);
    line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
  }
}

/**
 * Draw the bounding box of a pose. For example, for a whole person standing
 * in an image, the bounding box will begin at the nose and extend to one of
 * ankles
 */
function drawBoundingBox(boundingBox, pid, index) {
  fill(0,255,255);

  if (!guiState.trackingEnable) {
    textSize(20);
    text(`Person ${pid}`, boundingBox.minX, boundingBox.minY);
    // text(` [ ${pid} ]`, boundingBox.minX-15, boundingBox.minY);
  }
  noFill();
  strokeWeight(3);
  stroke(255, 0, 255);
  rect(
      boundingBox.minX, boundingBox.minY, boundingBox.maxX - boundingBox.minX,
      boundingBox.maxY - boundingBox.minY);
}
