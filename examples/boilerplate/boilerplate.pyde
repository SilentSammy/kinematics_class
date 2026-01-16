# 3D Visualization Boilerplate

angleX = 0
angleY = 0
zoom = 20

def setup():
    fullScreen(P3D)
    lights()

def draw():
    background(255)
    translate(width/2, height/2, 0)
    scale(zoom)
    
    rotateX(angleX)
    rotateY(angleY)
    rotateX(PI/2)
    rotateZ(PI/2)
    
    # Reference frame (RGB axes)
    strokeWeight(0.1)
    stroke(255, 0, 0)
    line(0, 0, 0, 10, 0, 0)
    stroke(0, 255, 0)
    line(0, 0, 0, 0, 10, 0)
    stroke(0, 0, 255)
    line(0, 0, 0, 0, 0, 10)
    
    # Your objects here
    stroke(0)
    fill(200, 100, 100)
    box(4, 4, 4)

def mouseDragged():
    global angleX, angleY
    angleY += (mouseX - pmouseX) * 0.01
    angleX -= (mouseY - pmouseY) * 0.01

def mouseWheel(event):
    global zoom
    zoom -= event.getCount() * 2  # Additive for smoother trackpad
    zoom = max(1, min(100, zoom))
