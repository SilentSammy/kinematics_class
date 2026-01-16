from text_helpers import Text, queue_text, draw_queued_text
import random

def setup():
    fullScreen(P3D)
    lights()

def draw():
    pushMatrix()
    setupScene()
    
    # Call the current slide function
    slides[current_slide]()
    
    popMatrix()
    
    # Draw all queued text in screen space
    slide_counter = "Slide {}/{}".format(current_slide + 1, len(slides))
    Text(slide_counter, 24, 0.01, 0.94, col=(128, 128, 128)).draw()
    Text("Use arrow keys to navigate slides", 20, 0.01, 0.97, col=(128, 128, 128)).draw()
    draw_queued_text()

# SLIDES
def sectionTitle(title, subtitle=""):
    queue_text(Text(title, 48, 0.5, 0.45, col=(0, 0, 0), h_align=CENTER, v_align=CENTER))
    if subtitle:
        queue_text(Text(subtitle, 32, 0.5, 0.55, col=(100, 100, 100), h_align=CENTER, v_align=CENTER))

def missingAxisQuestions():
    def question(q_idx, rotX, rotY, missing_idx):
        queue_text(Text("{}. Where will the missing axis point?".format(q_idx), 32, 0.05, 0.05, col=(0, 0, 0)))
        # Rotate view to throw off the student
        rotateX(rotX)
        rotateY(rotY)
        # Reference frame - hide the specified axis
        drawLines([axes[i] for i in range(3) if i != missing_idx])

    def answer(q_idx, rotX, rotY):
        queue_text(Text("{}. Answer".format(q_idx), 32, 0.05, 0.05))
        # Same rotation as question
        rotateX(rotX)
        rotateY(rotY)
        drawAxes()

    # Generate random questions
    axis_pool = [0, 1, 2]
    random.shuffle(axis_pool)
    
    questions = []
    
    # First 2 questions: random missing axis, no rotation
    for i in range(2):
        missing_idx = axis_pool[i]
        questions.append((0, 0, missing_idx))
    
    # Next 3 questions: completely random
    random.shuffle(axis_pool)
    for i in range(3):
        rotX = random.uniform(-PI/2, PI/2)
        rotY = random.uniform(-PI/2, PI/2)
        missing_idx = axis_pool[i % 3]  # Cycle through pool if needed
        questions.append((rotX, rotY, missing_idx))

    # Generate 6 functions: 3 missing axis questions and 3 answers
    slide_funcs = []
    slide_funcs.append(lambda: sectionTitle("Missing Axis Questions", "Identify the missing coordinate axis"))
    
    for i, (rotX, rotY, missing_idx) in enumerate(questions):
        q_idx = i + 1
        # Use default arguments to capture loop variables (avoid late binding)
        slide_funcs.append(lambda rx=rotX, ry=rotY, mi=missing_idx, qi=q_idx: question(qi, rx, ry, mi))
        slide_funcs.append(lambda rx=rotX, ry=rotY, qi=q_idx: answer(qi, rx, ry))
    
    return slide_funcs

# SLIDE STUFF
slides = []
slides.extend(missingAxisQuestions())

current_slide = 0

# DRAWING HELPERS
def drawLines(lines):
    for weight, col, line_coords in lines:
        strokeWeight(weight)
        stroke(*col)
        line(*line_coords)

def drawAxes():
    drawLines(axes)

axes = [
    (0.1, (255, 0, 0), (0, 0, 0, 10, 0, 0)),  # X-axis: red, goes to (10,0,0)
    (0.1, (0, 255, 0), (0, 0, 0, 0, 10, 0)),  # Y-axis: green, goes to (0,10,0)
    (0.1, (0, 0, 255), (0, 0, 0, 0, 0, 10)),  # Z-axis: blue, goes to (0,0,10)
]

# SCENE STUFF
angleX = -0.2
angleY = 0.2
zoom = 20

def setupScene():
    background(255)
    translate(width/2, height/2, 0)
    scale(zoom)
    
    rotateX(angleX)
    rotateY(angleY)
    rotateX(PI/2)
    rotateZ(PI/2)

# EVENTS
def mouseDragged():
    global angleX, angleY
    angleY += (mouseX - pmouseX) * 0.01
    angleX -= (mouseY - pmouseY) * 0.01

def mouseWheel(event):
    global zoom
    zoom -= event.getCount() * 2  # Additive for smoother trackpad
    zoom = max(1, min(100, zoom))

def keyPressed():
    global current_slide
    if key == CODED:
        if keyCode == RIGHT or keyCode == DOWN:
            current_slide = min(current_slide + 1, len(slides) - 1)
        elif keyCode == LEFT or keyCode == UP:
            current_slide = max(current_slide - 1, 0)
