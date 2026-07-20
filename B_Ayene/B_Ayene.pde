/**
 * B_Ayene — dot-matrix mirror
 *
 * I draw my head (from a_sense.py) as colored dots on black.
 * Wekinator sends the mood number; each mood has its own color and motion.
 *
 * I do not open the webcam here. On macOS only one program can use the camera
 * at a time; when a_sense.py was running, Processing showed a black feed. The
 * fix: Python sends the portrait over OSC and I draw from that grid instead.
 *
 * osc port 12000:
 *   /wek/outputs  → mood 1-4
 *   /ayene/grid   → face brightness blob from python
 */
import oscP5.*;
import netP5.*;

OscP5 osc;

int mood = 0;
float phase = 0;
float avgBright = 0;
int gridFrames = 0;
int lastGridMs = 0;

final int GRID_W = 96;
final int GRID_H = 72;
float[] grid = new float[GRID_W * GRID_H];

String[] names = {"Neutral", "Happy", "Surprised", "Focused"};
color[] cols = {
  color( 70, 150, 255),   // neutral  — sky blue
  color(255, 200,   0),   // happy    — gold
  color(  0, 255, 240),   // surprised — cyan
  color(210,  50, 255)    // focused  — purple
};
color[] glow = {
  color(140, 200, 255),
  color(255, 255,  80),
  color( 80, 255, 255),
  color(255, 120, 255)
};

void settings() {
  size(1280, 720, P2D);
  pixelDensity(1);
}

void setup() {
  // listen on port 12000 for Wekinator mood + Python portrait grid
  OscProperties props = new OscProperties();
  props.setListeningPort(12000);
  props.setDatagramSize(16384);   // big enough for /ayene/grid byte blob
  osc = new OscP5(this, props);
  ellipseMode(CENTER);
  rectMode(CENTER);
  noStroke();
}

// pick dot color — brighter face pixel = stronger color
color dotColor(float bri, boolean bright) {
  color c = bright ? glow[mood] : cols[mood];
  float g = 0.72 + bri * 0.5;
  return color(min(255, red(c)*g), min(255, green(c)*g), min(255, blue(c)*g));
}

// default round dot with a soft glow rim
void drawCircle(float x, float y, float size, float bri) {
  noStroke();
  fill(dotColor(bri, true));
  ellipse(x, y, size * 1.18, size * 1.18);
  fill(dotColor(bri, false));
  ellipse(x, y, size, size);
}

/*
 * mood visuals — this is the part you show in the presentation:
 *
 *   0 neutral   — calm blue circles, no motion
 *   1 happy     — gold circles wiggle like a wave (sin/cos offset)
 *   2 surprised — cyan circles pulse bigger/smaller (like a gasp)
 *   3 focused   — purple squares, no extra effects
 *
 * surprised is easy to change here when you have your idea.
 */
void drawMoodDot(float x, float y, float size, float bri, int gx, int gy) {

  if (mood == 0) {
    drawCircle(x, y, size, bri);
    return;
  }

  if (mood == 1) {
    // happy: dots vibrate in a wave pattern
    float wave = sin(phase * 4 + y * 0.05) * 3;
    float wave2 = cos(phase * 4 + x * 0.05) * 3;
    drawCircle(x + wave, y + wave2, size * 1.05, bri);
    return;
  }

  if (mood == 2) {
    // surprised: dots pulse in and out — staggered so it ripples across the face
    float pulse = 1.0 + sin(phase * 6 + (gx + gy) * 0.12) * 0.38;
    drawCircle(x, y, size * pulse, bri);
    return;
  }

  // focused: sharp purple squares only (no scan line)
  fill(dotColor(bri, false));
  rect(x, y, size * 0.9, size * 0.9);
}

void draw() {
  phase += 0.1;
  background(0);

  // show a hint until a_sense.py sends the first grid
  if (gridFrames == 0 || millis() - lastGridMs > 2000) {
    fill(200);
    textAlign(CENTER, CENTER);
    textSize(18);
    text("waiting for a_sense.py...", width/2, height/2);
    drawHUD();
    return;
  }

  float cellW = (float) width  / GRID_W;
  float cellH = (float) height / GRID_H;
  float dotMax = min(cellW, cellH) * 0.95;
  float sum = 0;
  int dots = 0;

  // walk the 96x72 grid — skip dark pixels, draw a dot for each bright one
  for (int gy = 0; gy < GRID_H; gy++) {
    for (int gx = 0; gx < GRID_W; gx++) {
      float bri = grid[gy * GRID_W + gx];
      sum += bri;
      bri = pow(bri, 0.65);
      if (bri < 0.07) continue;

      float size = bri * dotMax;
      float x = gx * cellW + cellW * 0.5;
      float y = gy * cellH + cellH * 0.5;
      dots++;

      drawMoodDot(x, y, size, bri, gx, gy);
    }
  }
  avgBright = (dots > 0) ? sum / dots : 0;
  drawHUD();
}

void drawHUD() {
  fill(0, 180);
  rect(width/2, height - 16, width, 32);
  fill(cols[mood]);
  rect(8, height - 16, 10, 22);
  fill(255);
  textAlign(LEFT, CENTER);
  textSize(14);
  text("MOOD: " + names[mood] + "  (" + (mood+1) + "/4)   bri: " +
       nf(avgBright, 1, 2) + "   fps: " + int(frameRate), 24, height - 16);
}

void oscEvent(OscMessage msg) {
  // mood from Wekinator: class 1-4, we store as 0-3
  if (msg.checkAddrPattern("/wek/outputs")) {
    int c = (msg.typetag().charAt(0) == 'i')
      ? msg.get(0).intValue() : round(msg.get(0).floatValue());
    mood = constrain(c - 1, 0, 3);
    return;
  }
  // head portrait from a_sense.py — one byte per grid cell (0-255)
  if (msg.checkAddrPattern("/ayene/grid")) {
    byte[] raw = msg.get(0).blobValue();
    for (int i = 0; i < min(grid.length, raw.length); i++) {
      grid[i] = (raw[i] & 0xFF) / 255.0;
    }
    gridFrames++;
    lastGridMs = millis();
  }
}
