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
let w = 320;
let h = 240;
const trackStartTime = 0.001;


function setup() {
  canvas = createCanvas(w, h);
  canvas.parent('sketch-holder');

  video = createVideo(['assets/cup.mp4'], () => {
    video.noLoop();
    video.volume(0);
    video.showControls();
    video.play();
  });
  video.parent( 'video-holder' );
  video.size(width, height);
  video.onended( (elm) => {
    video.play();
    // ready = false;
    // setupTrackingAlgorithm();
  });

  setupGui();
  stats = new Stats();
  stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
  document.body.appendChild(stats.dom);
}


let dst = null;
let roiHist = null;
let hsv = null;
let frame = null;
let trackWindow = null;
let termCrit = null;
let trackBox = null;
function setupTrackingAlgorithm() {

  ready = false;
  
  if (dst != null) dst.delete();
  if (roiHist !== null) roiHist.delete();
  if (hsv != null) hsv.delete();
  if (frame != null) frame.delete();
  if (trackWindow != null) trackWindow = null;
  if (termCrit != null) null;
  if (trackBox != null) null;

  console.log('[setupTrackingAlgorithm] video.time', video.time());
  
  video.loadPixels();
  // console.log(video.loadedmetadata, video.pixels.length, video.pixels);
  

  // take first frame of the video
  frame = new cv.Mat(video.height, video.width, cv.CV_8UC4);
  frame.data.set(video.pixels);

  // hardcode the initial location of window
  if (params.resolution == '320x240') {

    trackWindow = new cv.Rect(150, 60, 63, 125);
    
  } else {
    
    const _x = 640*200/320;
    const _y = 480*60/240;
    const _w = 640*63/320;
    const _h = 480*150/240;
    trackWindow = new cv.Rect(_x, _y, _w, _h);

  }

  // set up the ROI for tracking
  const roi = frame.roi(trackWindow);
  const hsvRoi = new cv.Mat();
  cv.cvtColor(roi, hsvRoi, cv.COLOR_RGBA2RGB);
  cv.cvtColor(hsvRoi, hsvRoi, cv.COLOR_RGB2HSV);
  const mask = new cv.Mat();
  const lowScalar = new cv.Scalar(30, 30, 0);
  const highScalar = new cv.Scalar(180, 180, 180);
  const low = new cv.Mat(hsvRoi.rows, hsvRoi.cols, hsvRoi.type(), lowScalar);
  const high = new cv.Mat(hsvRoi.rows, hsvRoi.cols, hsvRoi.type(), highScalar);
  cv.inRange(hsvRoi, low, high, mask);
  roiHist = new cv.Mat();
  const hsvRoiVec = new cv.MatVector();
  hsvRoiVec.push_back(hsvRoi);
  cv.calcHist(hsvRoiVec, [0], mask, roiHist, [180], [0, 180]);
  cv.normalize(roiHist, roiHist, 0, 255, cv.NORM_MINMAX);

  // delete useless mats.
  roi.delete(); hsvRoi.delete(); mask.delete(); low.delete(); high.delete(); hsvRoiVec.delete();

  // Setup the termination criteria, either 10 iteration or move by atleast 1 pt
  termCrit = new cv.TermCriteria(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1);

  hsv = new cv.Mat(video.height, video.width, cv.CV_8UC3);
  dst = new cv.Mat();
  hsvVec = new cv.MatVector();
  hsvVec.push_back(hsv);
  trackBox = null;

}


let params;
let gui;
function setupGui() {
  let Parameters = function () {
    this.algorithm = 'MeanShift';
    this.source = 'video';
    this.resolution = '320x240';
    this.drawKeypoints = true;
    this.drawSkeleton = true;
  };

  params = new Parameters();
  gui = new dat.GUI();
  gui.add(params, 'algorithm', ['MeanShift', 'CamShift']).onChange( (value) => {
    video.stop();
    video.play();
    ready = false;
    setupTrackingAlgorithm(); 
  });
  
  gui.add(params, 'resolution', ['320x240', '640x480']).onChange( (value) => {
    if (value == '320x240') {
      w = 320;
      h = 240
    } else {
      w = 640;
      h = 480;
    }
    video.stop();
    video.play();
    ready = false;
    canvas.resize(w, h, true);
    video.size(w, h);
    setupTrackingAlgorithm();

    {
      // resize canvas resets strokes and fill colours
      // https://github.com/processing/p5.js/issues/905
      colorMode(RGB, 255);
      fill(0);
      stroke(255);
    }
  });
}

let ready = false;
function cvReady() {
  if (!cv || !cv.loaded || video.time() < trackStartTime) return false;
  if (ready) return true;
  setupTrackingAlgorithm();
  ready = true;
  return true;
}

function draw() {
  stats.begin();

  clear();
  video.loadPixels();
  // image(video, 0, 0, width, height);

  if (video.pixels.length > 0) {
    if (cvReady()) {

      try {

        // start processing.
        frame.data.set(video.pixels);
        cv.cvtColor(frame, hsv, cv.COLOR_RGBA2RGB);
        cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);
        cv.calcBackProject(hsvVec, [0], roiHist, dst, [0, 180], 1);

        if (params.algorithm == 'MeanShift') {
          // Apply meanshift to get the new location
          // and it also returns number of iterations meanShift took to converge,
          // which is useless in this demo.
          [, trackWindow] = cv.meanShift(dst, trackWindow, termCrit);
  
          // Draw it on image
          let [x, y, w, h] = [trackWindow.x, trackWindow.y, trackWindow.width, trackWindow.height];
          // cv.rectangle(frame, new cv.Point(x, y), new cv.Point(x + w, y + h), [255, 0, 0, 255], 2);
  
          noFill();
          stroke(0, 255, 0);
          strokeWeight(2);
          rect(x,y,w,h);
          
        } else {
          // apply camshift to get the new location
          [trackBox, trackWindow] = cv.CamShift(dst, trackWindow, termCrit);

          // Draw it on image
          let pts = cv.rotatedRectPoints(trackBox);

          noFill();
          stroke(0, 0, 255);
          strokeWeight(2);
          line(pts[0].x, pts[0].y, pts[1].x, pts[1].y);
          line(pts[1].x, pts[1].y, pts[2].x, pts[2].y);
          line(pts[2].x, pts[2].y, pts[3].x, pts[3].y);
          line(pts[3].x, pts[3].y, pts[0].x, pts[0].y);
        }
        
      } catch (err) {
        console.error(err);
        
      }

    }
  }
  video.updatePixels();

  stats.end();
}