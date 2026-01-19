def setup():
    fullScreen(P3D)

def drawScene():
    background(255)
    
    # Set up lighting
    lights()
    directionalLight(180, 180, 180, -0.5, 0.5, -1)
    
    translate(width/2, height/2, 0)
    scale(zoom)
    
    rotateX(angleX)
    rotateY(angleY)
    rotateX(PI/2)
    rotateZ(PI/2)
    
def drawAxes():
    # Reference frame (RGB axes)
    pushStyle()
    strokeWeight(1)
    stroke(255, 0, 0)
    line(0, 0, 0, 100, 0, 0)
    stroke(0, 255, 0)
    line(0, 0, 0, 0, 100, 0)
    stroke(0, 0, 255)
    line(0, 0, 0, 0, 0, 100)
    popStyle()

def draw():
    drawScene()
    
    # Your objects here
    strokeWeight(0)
    fill(101, 50, 13)
    
    # tree trunk
    box(10, 10, 100)

    # Canopy
    pushMatrix()
    translate(0, 0, 50)
    fill(0, 255, 0)
    sphere(30)

    translate(0, 15, -15)
    sphere(20)
    translate(0, -30, 0)
    sphere(20)
    translate(0, 0, 15)
    sphere(20)
    translate(0, 30, 0)
    sphere(20)
    popMatrix()

    # Branch 1
    translate(0, 0, -5)
    fill(101, 50, 13) # brown
    pushMatrix()
    rotateX(PI/4)
    translate(0, 0, 10)
    box(3, 3, 20)
    translate(0, 0, 10)
    fill(0, 255, 0) # green
    sphere(7)
    popMatrix()

    # Branch 2 (mirror)
    pushMatrix()
    fill(101, 50, 13) # brown
    rotateX(-PI/4)
    translate(0, 0, 10)
    box(3, 3, 20)
    translate(0, 0, 10)
    fill(0, 255, 0) # green
    sphere(7)
    popMatrix()
    # drawAxes()

# Camera controls
angleX = 0
angleY = 0
zoom = 2

def mouseDragged():
    global angleX, angleY
    angleY += (mouseX - pmouseX) * 0.01
    angleX -= (mouseY - pmouseY) * 0.01

def mouseWheel(event):
    global zoom
    zoom -= event.getCount()  # Additive for smoother trackpad
    zoom = max(1, min(10, zoom))
