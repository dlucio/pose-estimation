// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
PoseNet example using p5.js
=== */

let video;
let poseNet;
let stats;
let poses = [];
let w = 640;
let h = 480;


function setup() {
  createCanvas(w, h);
  video = createCapture(VIDEO);
  video.size(width, height);

  // Create a new poseNet method with a single detection
  poseNet = ml5.poseNet(video, modelReady);
  // This sets up an event that fills the global variable "poses"
  // with an array every time new poses are detected
  poseNet.on('pose', function (results) {
    poses = results;
  });
  // Hide the video element, and just show the canvas
  video.hide();
  
  setupGui();
  stats = new Stats();
  stats.showPanel( 0 ); // 0: fps, 1: ms, 2: mb, 3+: custom
  document.body.appendChild( stats.dom );
}

function modelReady() {

  select('#status').html('Model Loaded');
}

let params;
let gui;
function setupGui() {
  let Parameters = function() {
    this.blurRadius = 5.0;
    this.threshold = 127.5;
    this.showThresholded = false;
    this.drawKeypoints = true;
    this.drawSkeleton = true;
  };

  params = new Parameters();
  gui = new dat.GUI();
  gui.add(params, 'blurRadius', 1.0, 10.0).step(0.1);
  gui.add(params, 'threshold', 0, 255).step(0.1);
  gui.add(params, 'showThresholded');
  gui.add(params, 'drawKeypoints');
  gui.add(params, 'drawSkeleton');
}

let captureMat, gray, blurred, thresholded;
let contours, hierarchy;
function cvSetup() {
  captureMat = new cv.Mat(h, w, cv.CV_8UC4);
  gray = new cv.Mat(h, w, cv.CV_8UC1);
  blurred = new cv.Mat(h, w, cv.CV_8UC1);
  thresholded = new cv.Mat(h, w, cv.CV_8UC1);
}

let ready = false;
function cvReady() {
  if (!cv || !cv.loaded) return false;
  if (ready) return true;
  cvSetup();
  ready = true;
  return true;
}

function draw() {
  const showThresholded = params.showThresholded;

  stats.begin();

  if (cvReady()) {
    video.loadPixels();
    if (video.pixels.length > 0) {
      captureMat.data.set(video.pixels);

      const blurRadius = params.blurRadius;
      const threshold = params.threshold;

      cv.cvtColor(captureMat, gray, cv.COLOR_RGBA2GRAY, 0);
      cv.blur(gray, blurred, new cv.Size(blurRadius, blurRadius), new cv.Point(-1, -1), cv.BORDER_DEFAULT);
      cv.threshold(blurred, thresholded, threshold, 255, cv.THRESH_BINARY);

      if (showThresholded) {
        const src = thresholded.data;
        let  dst = video.pixels;
        const n = src.length;
        let j = 0;
        for (let i = 0; i < n; i++) {
          dst[j++] = src[i];
          dst[j++] = src[i];
          dst[j++] = src[i];
          dst[j++] = 255;
        }
        video.updatePixels();
      }

      if (contours) {
        contours.delete();
      }
      if (hierarchy) {
        hierarchy.delete();
      }
      contours = new cv.MatVector();
      hierarchy = new cv.Mat();
      cv.findContours(thresholded, contours, hierarchy, 3, 2, new cv.Point(0, 0));
    }
  }


  image(video, 0, 0, width, height);

  if (contours && !showThresholded) {
    
    noStroke();
    for (let i = 0; i < contours.size(); i++) {
      fill(0, 0, 255, 128);
      let contour = contours.get(i);

      beginShape();
      let k = 0;
      for (let j = 0; j < contour.total(); j++) {
        const x = contour.data[k++];
        const y = contour.data[k++];
        vertex(x, y);
      }
      endShape(CLOSE);

      noFill();
      stroke(255, 255, 255)
      const box = cv.boundingRect(contour);
      rect(box.x, box.y, box.width, box.height);

      // these aren't working right now:
      // https://github.com/ucisysarch/opencvjs/issues/30
      //            var minAreaRect = cv.minAreaRect(contour);
      //            var minAreaEllipse = cv.ellipse1(contour);
      //            var fitEllipse = cv.fitEllipse(contour);
    }

  }
  
  // We can call both functions to draw all keypoints and the skeletons
  if (params.drawKeypoints) {
    drawKeypoints();
  }
  
  if (params.drawSkeleton) {
    drawSkeleton();
  }
  
  stats.end();
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints() {
  // Loop through all the poses detected
  for (let i = 0; i < poses.length; i++) {
    // For each pose detected, loop through all the keypoints
    let pose = poses[i].pose;
    for (let j = 0; j < pose.keypoints.length; j++) {
      // A keypoint is an object describing a body part (like rightArm or leftShoulder)
      let keypoint = pose.keypoints[j];
      // Only draw an ellipse is the pose probability is bigger than 0.2
      if (keypoint.score > 0.2) {
        fill(255, 0, 0);
        noStroke();
        ellipse(keypoint.position.x, keypoint.position.y, 10, 10);
      }
    }
  }
}

// A function to draw the skeletons
function drawSkeleton() {
  // Loop through all the skeletons detected
  for (let i = 0; i < poses.length; i++) {
    let skeleton = poses[i].skeleton;
    // For every skeleton, loop through all body connections
    for (let j = 0; j < skeleton.length; j++) {
      let partA = skeleton[j][0];
      let partB = skeleton[j][1];
      stroke(255, 0, 0);
      line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
    }
  }
}
