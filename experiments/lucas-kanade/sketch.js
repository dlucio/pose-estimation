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
let stats;
let w = 640;
let h = 480;
const trackStartTime = 0.009;


function setup() {
  canvas = createCanvas(w, h);
  canvas.parent('sketch-holder');

  video = createVideo(['assets/box.mp4'], () => {
    video.noLoop();
    video.volume(0);
    video.showControls();
    video.play();
  });
  video.parent( 'video-holder' );
  video.size(width, height);
  video.onended( (elm) => {
    video.play();
    ready = false;
    // setupTrackingAlgorithm();
  });

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
function setupTrackingAlgorithm() {

  ready = false;
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
  winSize = new cv.Size(params.winSize, params.winSize);
  maxLevel = params.maxLevel;
  criteria = new cv.TermCriteria(
    cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 
    params.criteriaMaxCount, params.criteriaEpsilon);

  // create some random colors
  color = [];
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
  p0 = new cv.Mat();
  none = new cv.Mat();
  cv.goodFeaturesToTrack(oldGray, p0, maxCorners, qualityLevel, minDistance, none, blockSize);

  frame = new cv.Mat(video.height, video.width, cv.CV_8UC4);
  frameGray = new cv.Mat();
  p1 = new cv.Mat();
  st = new cv.Mat();
  err = new cv.Mat();

}

function reloadAlgorithm() {
  ready = false;
  video.stop();
  video.play();
  setupTrackingAlgorithm();
}

let params;
let gui;
function setupGui() {
  let Parameters = function () {
    this.source = 'video';
    this.resolution = '640x480';
    this.videoOpacity= 1.0;
    this.winSize = 15;
    this.maxLevel = 2;
    this.criteriaMaxCount = 10;
    this.criteriaEpsilon = 0.03;
  };

  params = new Parameters();
  gui = new dat.GUI();
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

  gui.add(params, 'videoOpacity', 0.0, 1.0);
  gui.add(params, 'winSize', 12, 30).step(1.0).onChange( () => reloadAlgorithm() );
  gui.add(params, 'maxLevel', 1, 5 ).step(1.0).onChange( () => reloadAlgorithm() );
  let criteriaFolder = gui.addFolder('Criteria');
  criteriaFolder.add(params, 'criteriaMaxCount', 9, 20).name('max count').step(1.0).onChange( () => reloadAlgorithm() );
  criteriaFolder.add(params, 'criteriaEpsilon', 0.01, 0.1).name('epsilon').step(0.01).onChange( () => reloadAlgorithm() );
  criteriaFolder.open();
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

  video.style('opacity', params.videoOpacity);

  if (video.pixels.length > 0) {
    if (cvReady()) {

      try {
        frame.data.set(video.pixels);
        cv.cvtColor(frame, frameGray, cv.COLOR_RGBA2GRAY);
        // calculate optical flow
        cv.calcOpticalFlowPyrLK(oldGray, frameGray, p0, p1, st, err, winSize, params.maxLevel, criteria);
        
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

          fill(...color[i]);
          ellipse(goodNew[i].x, goodNew[i].y, 5);
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
        ready=false;
      }

    }
  }
  video.updatePixels();

  stats.end();
}