// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
PoseNet example using p5.js
=== */

let canvas;
let video;
let poseNet;
let stats;
let poses = [];
let interval;

const videoSrc = { 
  'video 1': ['../../assets/u2_640x360.mp4'], 
  'video 2': ['../../assets/frevo_640x360.mp4'],
  'video 3': ['../../assets/pomplamoose_640x360.mp4'],
  'video 4': ['../../assets/dancing_640x360.mp4'],
  'video 5': ['../../assets/soccer_video_640x360.mp4'],
}


let isModelReady = false;

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
  // video = createVideo(videoSrc['video 1'], () => {
  //   video.loop();
  //   video.volume(0);
  //   video.pause();
  //   video.showControls();
  //   const w = video.width;
  //   const h = video.height;
  //   let canvas = createCanvas(w, h);
  //   canvas.parent('sketch-holder');
  // });
  
  video = createCapture(VIDEO, () => {
    // NOTE: hardcode canvas size for my camera
    // I don't know why video withxheight == 300x150;
    let canvas = createCanvas(640, 480);
    canvas.parent('sketch-holder');
    console.log('video size', video.width, video.height);
    
    
  });
  video.parent( 'video-holder' );

  setupPoseNet();
  setupGui();
  
  stats = new Stats();
  stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
  document.body.appendChild(stats.dom);
}


let winSize = null;
let maxLevel = null;
let criteria = null;
let color = [];
let frame = null;
let frameGray = null;
let oldFrame = null;
let oldGray = null;
let p0 = null;
let p1 = null;
let st = null;
let err = null;
let none = null;
const trackStartTime = 0.001;
function setupTrackingAlgorithm() {

  if (poses.length < 1) {
    return false;
  }

  if (winSize !== null) { delete winSize; winSize = null; }
  if (criteria !== null) { delete criteria; criteria = null; }
  if (color.length > 0) { color = [] };
  if (frame !== null) { frame.delete(); frame = null; }
  if (frameGray !== null) { frameGray.delete(); frameGray = null; }
  if (oldFrame !== null) { oldFrame.delete(); oldFrame = null; }
  if (oldGray !== null) { oldGray.delete(); oldGray = null; }
  if (p0 !== null) { p0.delete(); p0 = null; }
  if (p1 !== null) { p1.delete(); p1 = null; }
  if (st !== null) { st.delete(); st = null; }
  if (err !== null) { err.delete(); err = null; }
  if (none !== null) { none.delete(); none = null; }
  
  // parameters for ShiTomasi corner detection
  let [maxCorners, qualityLevel, minDistance, blockSize] = [30, 0.3, 7, 7];

  // parameters for lucas kanade optical flow
  const ws = guiState.tracking.winSize;
  winSize = new cv.Size(ws, ws);
  maxLevel = guiState.tracking.maxLevel;
  criteria = new cv.TermCriteria(
    cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 
    guiState.tracking.criteria.maxCount, 
    guiState.tracking.criteria.epsilon);

  // create some random colors
  color = [];
  const numOfKeypoints = 17;
  for (let i = 0; i < maxCorners; i++) {
      color.push(new cv.Scalar(parseInt(Math.random()*255), parseInt(Math.random()*255),
                              parseInt(Math.random()*255), 255));
  }


  video.loadPixels();

  // take first frame and find corners in it
  oldFrame = new cv.Mat(video.height, video.width, cv.CV_8UC4);
  oldFrame.data.set(video.pixels)
  oldGray = new cv.Mat();
  cv.cvtColor(oldFrame, oldGray, cv.COLOR_RGB2GRAY);
  
  
  const person = poses[0];
  const points = [];
  person.pose.keypoints.forEach((keypoint,i) => {
    if (keypoint.score > minPartConfidence) {
      const x = Math.min(keypoint.position.x, width);
      const y = Math.min(keypoint.position.y, height);
      points.push( {x:x, y:y} );
    }
  });
  
  p0 = new cv.Mat(points.length, 1, cv.CV_32FC2);
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    
    p0.data32F[i*2+0] = p.x;
    p0.data32F[i*2+1] = p.y;
  }

  frame = new cv.Mat(video.height, video.width, cv.CV_8UC4);
  frameGray = new cv.Mat();
  p1 = new cv.Mat();
  st = new cv.Mat();
  err = new cv.Mat();

  return true;
}

let ready = false;
function cvReady() {

  const inputLoaded = (guiState.source == 'webcam') 
      ? isModelReady && (poses.length > 0) 
      : (video.time() > trackStartTime) && isModelReady && (poses.length > 0);

  
  if (!cv || !cv.loaded || !inputLoaded) return false;
  if (ready) return true;
  setupTrackingAlgorithm();
  ready = setupTrackingAlgorithm();
  return ready;
}


function setupPoseNet() {
  isModelReady = false;
  ready = false; 
  
  delete poses;
  poses = null;
  poses = [];

  if (poseNet != null) {
    poseNet.net.dispose();
    delete poseNet;
    poseNet = null;
  }
  // Create a new poseNet method with a single detection
  poseNet = ml5.poseNet(video, config,  modelReady);

  // This sets up an event that fills the global variable "poses"
  // with an array every time new poses are detected
  poseNet.on('pose', function (results) {
    delete poses;
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
  console.log("model loaded");

}


const tryResNetButtonName = 'tryResNetButton';
const tryResNetButtonText = '[New] Try ResNet50';

const guiState = {
  algorithm: 'multi-pose',
  source: 'webcam', // video n or webcam
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
    showBoundingBox: false,
  },
  estimatePoseEnable: true,
  trackingEnable: true,
  tracking: {
    useInterval: false,
    interval: 2,
    restart: () => console.log('tracking is restarting'),
    criteria: {
      maxCount: 10,
      epsilon: 0.03,
    },
    winSize: 15,
    maxLevel: 2,
  },
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
          
        });

      } else {

        video = createCapture(VIDEO, () => {
          const w = video.width;
          const h = video.height;
          console.log('Why WxH =', w,h);
          
          // NOTE: hardcode canvas size for my camera
          // I don't know why video withxheight == 300x150;
          resizeCanvas(640, 480);

        });
        
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

  let tracking = gui.addFolder('Tracking');
  tracking.add(guiState.tracking, 'useInterval')
    .onChange( (value) => {

      if (typeof(interval) !== 'undefined') {
        clearInterval(interval);
        console.log("cleaning interval");
                  
      }

      if (value) {
        
        interval = setInterval( () => {
          if (guiState.tracking.useInterval) {
            isModelReady = false;
            setupTrackingAlgorithm();
            isModelReady = true;
          } 
        }, guiState.tracking.interval*1000);

      }
      
    });

  tracking.add(guiState.tracking, 'interval', 1, 10)
    .onChange( value => {

      if (typeof(interval) !== 'undefined') {
        clearInterval(interval);
        console.log("cleaning interval");
                  
      }

      interval = setInterval( () => {
        if (guiState.tracking.useInterval) {
          ready = false;
        }
      }, value*1000);
      
    });

  tracking.add(guiState.tracking, 'winSize', 12, 30).step(1).onChange( () => ready=false );
  tracking.add(guiState.tracking, 'maxLevel', 1, 5).step(1).onChange( () => ready=false );
  let criteriaFolder = tracking.addFolder('LK Criteria');
  criteriaFolder.add(guiState.tracking.criteria, 'maxCount', 9, 20).step(1).onChange( () => ready=false );
  criteriaFolder.open();
  criteriaFolder.add(guiState.tracking.criteria, 'epsilon', 0.01, 0.1).step(0.01).onChange( () => ready=false );
  tracking.add(guiState.tracking, 'restart').name('press to restart tracking').onChange( () => ready=false );
  tracking.open();

  {
    // resize canvas resets strokes and fill colours
    // https://github.com/processing/p5.js/issues/905
    colorMode(RGB, 255);
    fill(0);
    stroke(255);
  }

}

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


  switch (guiState.algorithm) {
    case 'single-pose':
      minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
      minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;
      break;
    case 'multi-pose':
      minPoseConfidence = +guiState.multiPoseDetection.minPoseConfidence;
      minPartConfidence = +guiState.multiPoseDetection.minPartConfidence;
      break;
  }
}


function draw() {
  stats.begin();

  clear();
  video.style('opacity', guiState.output.videoOpacity);


  video.loadPixels();
  if (video.pixels.length > 0) {
    if (cvReady()) {

      try {
        frame.data.set(video.pixels);
        cv.cvtColor(frame, frameGray, cv.COLOR_RGBA2GRAY);
        // calculate optical flow
        cv.calcOpticalFlowPyrLK(oldGray, frameGray, p0, p1, st, err, winSize, maxLevel, criteria);
        
        // select good points
        let goodNew = [];
        let goodOld = [];
        for (let i = 0; i < st.rows; i++) {
            if (st.data[i] === 1) {
                goodNew.push(new cv.Point(p1.data32F[i*2], p1.data32F[i*2+1]));
                goodOld.push(new cv.Point(p0.data32F[i*2], p0.data32F[i*2+1]));
            }
        }

        // draw the tracks
        for (let i = 0; i < goodNew.length; i++) {
          stroke(...color[i]);
          strokeWeight(2);
          line(goodNew[i].x, goodNew[i].y, goodOld[i].x, goodOld[i].y);

          const r = guiState.source == 'webcam' ? 10 : 5;
          fill(...color[i]);
          ellipse(goodNew[i].x, goodNew[i].y, r);
        }

         // now update the previous frame and previous points
         frameGray.copyTo(oldGray);
         p0.delete(); p0 = null;
         p0 = new cv.Mat(goodNew.length, 1, cv.CV_32FC2);
         for (let i = 0; i < goodNew.length; i++) {
             p0.data32F[i*2] = goodNew[i].x;
             p0.data32F[i*2+1] = goodNew[i].y;
         }         

      } catch (error) {
        console.error(error);
      }

    }
  }
  video.updatePixels();



  
  if (guiState.estimatePoseEnable && isModelReady) {
    
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
      const r = guiState.source == 'webcam' ? 10 : 5;
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
    strokeWeight(guiState.source == 'webcam' ? 3 : 2);
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

  textSize(20);
  text(`Person ${pid}`, boundingBox.minX, boundingBox.minY);
  // text(` [ ${pid} ]`, boundingBox.minX-15, boundingBox.minY);
  
  noFill();
  strokeWeight(guiState.source == 'webcam' ? 3 : 2);
  stroke(255, 0, 255);
  rect(
      boundingBox.minX, boundingBox.minY, boundingBox.maxX - boundingBox.minX,
      boundingBox.maxY - boundingBox.minY);
}
